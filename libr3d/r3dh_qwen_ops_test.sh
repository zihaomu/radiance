#!/usr/bin/env bash
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
workspace=$(dirname -- "$repo")
mode=${1:---host}
if [[ "$mode" != --host && "$mode" != --gpu && "$mode" != --all ]]; then
    printf 'Usage: bash libr3d/r3dh_qwen_ops_test.sh --host|--gpu|--all\n' >&2
    exit 2
fi
image=freetoken:rocm10.1.0-transformers518-final-gh-proxy-20261026
binary=$workspace/build-r3d-bootstrap/r3dh_qwen_ops_check
logs=$workspace/results/r3d-iqr/implementation/logs
mkdir -p "$(dirname -- "$binary")" "$logs"
base=(docker run --rm --user 1001:1001 -v "$workspace:$workspace" -w "$repo")
compiler=("${base[@]}" --entrypoint /opt/rocm/llvm/bin/clang++ "$image")
flags=(-std=c++20 -O2 -fPIC -fno-fast-math -ffp-contract=off -fdenormal-fp-math=ieee
    -ffunction-sections -fdata-sections --offload-arch=gfx1100 -Iabi -Ilibr3d -Ilibref)
sources=(libr3d/r3dh_qwen_ops_quant.hip libr3d/r3dh_qwen_ops_router.hip
    libr3d/r3dh_qwen_ops_check.hip libr3d/r3d_moe_router_topk_bf16.hip
    libr3d/r3d_moe_scatter.hip libr3d/r3d_moe_gather.hip)
printf 'QWEN_BUILD_FLAGS %s\n' "${flags[*]}" > "$logs/r3dh_qwen_ops-build.log"
objects=()
for source in libref/ref_quant.cpp libref/ref_moe.cpp libref/ref_gemm.cpp; do
    name=${source##*/}
    object=${binary}_${name%.cpp}.o
    "${compiler[@]}" -std=c++20 -O2 -fPIC -fno-fast-math -ffp-contract=off \
        -fdenormal-fp-math=ieee -ffunction-sections -fdata-sections -Iabi -Ilibref \
        -c "$source" -o "$object" >> "$logs/r3dh_qwen_ops-build.log" 2>&1
    objects+=("$object")
done
"${compiler[@]}" "${flags[@]}" "${sources[@]}" "${objects[@]}" -Wl,--gc-sections -o "$binary" \
    >> "$logs/r3dh_qwen_ops-build.log" 2>&1
for forbidden in -ffast-math -ffinite-math-only; do
    if "${compiler[@]}" -std=c++20 "$forbidden" -x c++ -Iabi -Ilibr3d \
        -fsyntax-only libr3d/r3dh_qwen_ops.h > "$logs/r3dh_qwen_ops-policy.log" 2>&1; then
        printf 'QWEN_FAIL unsafe flag accepted: %s\n' "$forbidden" >&2
        exit 1
    fi
    grep -q 'Qwen ops require strict' "$logs/r3dh_qwen_ops-policy.log"
done
"${base[@]}" --entrypoint "$binary" "$image" --host > "$logs/r3dh_qwen_ops-host.log" 2>&1
grep 'QWEN_HOST_PASS' "$logs/r3dh_qwen_ops-host.log"
printf 'QWEN_BUILD_POLICY_PASS fastmath_rejected=1 finite_only_rejected=1\n'
if [[ "$mode" == --host ]]; then exit 0; fi
lock=/home/zihaomu/bigssd/oj_system/var/locks/gpu-renderD130.lock
if { flock -n -E 75 8 && "${base[@]}" --device /dev/kfd --device /dev/dri \
    --group-add 993 --group-add 44 --security-opt seccomp=unconfined \
    -e ROCR_VISIBLE_DEVICES=2 -e HIP_VISIBLE_DEVICES=0 \
    --entrypoint "$binary" "$image" --gpu; } 8<"$lock" > "$logs/r3dh_qwen_ops-gpu.log" 2>&1; then
    grep 'QWEN_.*PASS' "$logs/r3dh_qwen_ops-gpu.log"
else
    status=$?
    printf 'QWEN_GPU_EXIT status=%d (75 means production GPU2 lock busy)\n' "$status" >&2
    tail -12 "$logs/r3dh_qwen_ops-gpu.log" >&2
    exit "$status"
fi
