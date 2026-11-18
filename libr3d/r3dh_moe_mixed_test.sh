#!/usr/bin/env bash
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
workspace=$(dirname -- "$repo")
mode=${1:---host}
if [[ "$mode" != --host && "$mode" != --all ]]; then
    printf 'usage: bash libr3d/r3dh_moe_mixed_test.sh --host|--all [MODEL] [LIBREF]\n' >&2
    exit 2
fi
model=${2:-$workspace/models/qwen3.8-next-flash-iqr-v4.rad}
libref=${3:-$workspace/build-r3d-gfx1100/radiance_home/kernels/libref.so}
build=$workspace/build-r3d-bootstrap
logs=$workspace/results/r3d-iqr/implementation/logs
image=freetoken:rocm10.1.0-transformers518-final-gh-proxy-20261026
mkdir -p "$build" "$logs"
compiler=(docker exec -w "$repo" r3dh-moe-iqr-sdk /opt/rocm/llvm/bin/clang++)
flags=(-std=c++17 -O2 -ffp-contract=off -fno-fast-math --offload-arch=gfx1100 -Iabi -Ilibr3d -x hip)
sources=(libr3d/r3dh_moe_mixed_iqr.hip libr3d/r3dh_moe_mixed_bf16.hip
    libr3d/r3dh_moe_mixed_layout.cpp libr3d/r3dh_moe_iqr_layout.cpp
    libr3d/r3dh_moe_w4nl.hip libr3d/r3dh_moe_w4nl_layout.cpp)
{
    "${compiler[@]}" --version
    printf 'MIXED_FLAGS %s\n' "${flags[*]}"
    "${compiler[@]}" "${flags[@]}" libr3d/r3dh_moe_mixed_check.cpp libr3d/r3dh_moe_mixed_registry.cpp \
        "${sources[@]}" -o "$build/r3dh_moe_mixed_check"
    "${compiler[@]}" "${flags[@]}" libr3d/r3dh_moe_mixed_numeric_check.hip libr3d/r3dh_moe_iqr.hip \
        "${sources[@]}" -ldl -o "$build/r3dh_moe_mixed_numeric_check"
    "${compiler[@]}" -std=c++17 -O2 -Iabi -Ilibr3d libr3d/r3dh_moe_mixed_model_check.cpp \
        -o "$build/r3dh_moe_mixed_model_check"
} > "$logs/mixed-finish-build.log" 2>&1
docker exec r3dh-moe-iqr-sdk "$build/r3dh_moe_mixed_check" > "$logs/mixed-finish-host.log" 2>&1
docker exec r3dh-moe-iqr-sdk "$build/r3dh_moe_mixed_model_check" "$model" > "$logs/mixed-finish-model.log" 2>&1
docker exec -e OMP_NUM_THREADS=1 r3dh-moe-iqr-sdk "$build/r3dh_moe_mixed_numeric_check" \
    "$model" "$libref" --host > "$logs/mixed-finish-cpu-numeric.log" 2>&1
grep 'MIXED_SHAPE_PASS' "$logs/mixed-finish-host.log"
grep 'MIXED_MODEL_PASS' "$logs/mixed-finish-model.log"
grep 'MIXED_ALL_PASS' "$logs/mixed-finish-cpu-numeric.log"
if [[ "$mode" == --all ]]; then
    lock=/home/zihaomu/bigssd/oj_system/var/locks/gpu-renderD133.lock
    {
        flock -n -E 75 8
        docker run --rm --name r3dh-mixed-finish-gpu5 --user 1001:1001 \
            --device /dev/kfd --device /dev/dri --group-add 993 --group-add 44 \
            --security-opt seccomp=unconfined -e ROCR_VISIBLE_DEVICES=5 -e HIP_VISIBLE_DEVICES=0 \
            -e OMP_NUM_THREADS=1 -v "$workspace:$workspace" \
            --entrypoint "$build/r3dh_moe_mixed_numeric_check" "$image" "$model" "$libref" --gpu
    } 8<"$lock" > "$logs/mixed-finish-gpu5-numeric.log" 2>&1
    grep 'MIXED_GPU_ID\|MIXED_ALL_PASS' "$logs/mixed-finish-gpu5-numeric.log"
    { flock -n -E 75 8; printf 'MIXED_GPU5_LOCK_RELEASED\n'; } 8<"$lock" > "$logs/mixed-finish-lock.log"
    grep 'MIXED_GPU5_LOCK_RELEASED' "$logs/mixed-finish-lock.log"
fi
printf 'MIXED_FINISH_PASS mode=%s logs=%s\n' "$mode" "$logs"
