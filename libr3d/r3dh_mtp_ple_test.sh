#!/usr/bin/env bash
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
workspace=$(dirname -- "$repo")
image=freetoken:rocm10.1.0-transformers518-final-gh-proxy-20261026
build=$workspace/build-r3d-bootstrap
logs=$workspace/results/r3d-iqr/implementation/logs
mkdir -p "$build" "$logs"
binary=$build/r3dh_mtp_ple_check
base=(docker run --rm --user 1001:1001 -v "$workspace:$workspace" -w "$repo")
compiler=("${base[@]}" --entrypoint /opt/rocm/llvm/bin/clang++ "$image")
flags=(-std=c++20 -O2 -ffp-contract=off -fno-fast-math -ffunction-sections -fdata-sections -Iabi -Ilibr3d -Ilibref)
mode=${1:---host}
case "$mode" in --host|--all|--build|--model|--model-host) ;; *) exit 2 ;; esac
{
    "${compiler[@]}" --version
    objects=()
    for source in ref_hc ref_ple ref_elementwise ref_gemm ref_ngram ref_moe; do
        object=$build/r3dh_mtp_ple_$source.o
        "${compiler[@]}" "${flags[@]}" -c "libref/$source.cpp" -o "$object"
        objects+=("$object")
    done
    for source in layout shapes; do
        "${compiler[@]}" "${flags[@]}" -fPIC -c "libr3d/r3dh_mtp_ple_$source.cpp" -o "$build/r3dh_mtp_ple_$source.o"
    done
    port=$build/r3dh_mtp_ple_port.so
    "${compiler[@]}" "${flags[@]}" -fPIC -shared --offload-arch=gfx1100 libr3d/r3d_mtp_registry.cpp libr3d/r3dh_mtp_ple_native.hip libr3d/r3dh_mtp_ple_ngram_ids.hip libr3d/r3dh_mtp_ple_select.hip \
        "$build/r3dh_mtp_ple_layout.o" "$build/r3dh_mtp_ple_shapes.o" -o "$port"
    "${compiler[@]}" "${flags[@]}" --offload-arch=gfx1100 libr3d/r3dh_mtp_ple_check.hip \
        "$port" "${objects[@]}" -Wl,--gc-sections -Wl,-rpath,"$build" -o "$binary"
} > "$logs/r3dh_mtp_ple-build.log" 2>&1
printf 'MTP_PLE_BUILD_PASS gfx1100 strict_fp=1\n'
if [[ "$mode" == --build ]]; then exit 0; fi
"${base[@]}" --entrypoint "$binary" "$image" --host > "$logs/r3dh_mtp_ple-host.log" 2>&1
tail -1 "$logs/r3dh_mtp_ple-host.log"
"${base[@]}" --entrypoint "$binary" "$image" --inspect "$workspace/models/qwen3.8-next-flash-iqr-v4.rad" > "$logs/r3dh_mtp_ple-model-inspect.log" 2>&1
if [[ "$mode" == --host ]]; then exit 0; fi
if [[ "$mode" == --model-host ]]; then
    "${base[@]}" --entrypoint "$binary" "$image" --model-host "$workspace/models/qwen3.8-next-flash-iqr-v4.rad" > "$logs/r3dh_mtp_ple-model-host.log" 2>&1
    tail -1 "$logs/r3dh_mtp_ple-model-host.log"
    exit 0
fi
lock=/home/zihaomu/bigssd/oj_system/var/locks/gpu-renderD131.lock
gpu_args=(--gpu)
if [[ "$mode" == --model ]]; then gpu_args=(--model "$workspace/models/qwen3.8-next-flash-iqr-v4.rad"); fi
if { flock -n -E 75 8 && "${base[@]}" --device /dev/kfd --device /dev/dri --group-add 993 --group-add 44 \
    --security-opt seccomp=unconfined -e ROCR_VISIBLE_DEVICES=3 -e HIP_VISIBLE_DEVICES=0 \
    --entrypoint "$binary" "$image" "${gpu_args[@]}"; } 8<"$lock" > "$logs/r3dh_mtp_ple-gpu.log" 2>&1; then
    tail -1 "$logs/r3dh_mtp_ple-gpu.log"
else
    status=$?
    printf 'MTP_PLE_GPU_EXIT status=%d (75=production GPU3 lock busy; no wait)\n' "$status" >&2
    exit "$status"
fi
