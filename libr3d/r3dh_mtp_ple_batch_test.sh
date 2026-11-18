#!/usr/bin/env bash
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
workspace=$(dirname -- "$repo")
image=freetoken:rocm10.1.0-transformers518-final-gh-proxy-20261026
mode=${1:---all}
case "$mode" in --host|--all|--build) ;; *) exit 2 ;; esac
build=$workspace/build-r3d-bootstrap/mtp-ple-batch
mkdir -p "$build" "$workspace/results/r3d-iqr/implementation"
logs=$(mktemp -d "$workspace/results/r3d-iqr/implementation/mtp-ple-batch-XXXXXXXX")
printf 'BATCH_LOGS %s\n' "$logs"
binary=$build/r3dh_mtp_ple_check
base=(docker run --rm --user 1001:1001 -v "$workspace:$workspace" -w "$repo")
compiler=("${base[@]}" --entrypoint /opt/rocm/llvm/bin/clang++ "$image")
flags=(-std=c++20 -O2 -ffp-contract=off -fno-fast-math -ffunction-sections -fdata-sections -Iabi -Ilibr3d -Ilibref)
{
    "${compiler[@]}" --version
    objects=()
    for source in ref_hc ref_ple ref_elementwise ref_gemm ref_ngram ref_moe; do
        object=$build/$source.o
        "${compiler[@]}" "${flags[@]}" -c "libref/$source.cpp" -o "$object"
        objects+=("$object")
    done
    for source in layout shapes; do
        object=$build/$source.o
        "${compiler[@]}" "${flags[@]}" -fPIC -c "libr3d/r3dh_mtp_ple_$source.cpp" -o "$object"
        objects+=("$object")
    done
    "${compiler[@]}" "${flags[@]}" --offload-arch=gfx1100 libr3d/r3dh_mtp_ple_check.hip \
        libr3d/r3d_mtp_registry.cpp \
        libr3d/r3dh_mtp_ple_native.hip libr3d/r3dh_mtp_ple_ngram_ids.hip libr3d/r3dh_mtp_ple_select.hip \
        "${objects[@]}" -Wl,--gc-sections -o "$binary"
} > "$logs/build.log" 2>&1
sha256sum "$binary" "$repo"/libr3d/r3dh_mtp_ple* "$repo/libr3d/r3d_mtp_registry.cpp" > "$logs/source-sha256.txt"
printf 'BATCH_BUILD_PASS strict_fp=1\n'
if [[ "$mode" == --build ]]; then exit 0; fi
"${base[@]}" --entrypoint "$binary" "$image" --batch-host > "$logs/host.log" 2>&1
tail -1 "$logs/host.log"
if [[ "$mode" == --host ]]; then exit 0; fi
lock=/home/zihaomu/bigssd/oj_system/var/locks/gpu-renderD134.lock
if { flock -n -E 75 8 && "${base[@]}" --device /dev/kfd --device /dev/dri --group-add 993 --group-add 44 \
    --security-opt seccomp=unconfined -e ROCR_VISIBLE_DEVICES=6 -e HIP_VISIBLE_DEVICES=0 \
    --entrypoint "$binary" "$image" --batch "$workspace/models/qwen3.8-next-flash-iqr-v4.rad"; } 8<"$lock" > "$logs/gpu.log" 2>&1; then
    tail -1 "$logs/gpu.log"
else
    status=$?
    tail -8 "$logs/gpu.log" >&2
    printf 'BATCH_GPU_EXIT status=%d (75=GPU6 production lock busy; no wait)\n' "$status" >&2
    exit "$status"
fi
