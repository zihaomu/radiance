#!/usr/bin/env bash
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
workspace=$(dirname -- "$repo")
mode=${1:---host}
model=${2:-$workspace/models/qwen3.8-next-flash-iqr-v4.rad}
if [[ "$mode" != --host && "$mode" != --all ]]; then
    printf 'Usage: bash libr3d/r3dh_moe_w4nl_test.sh --host|--all [MODEL]\n' >&2
    exit 2
fi
image=freetoken:rocm10.1.0-transformers518-final-gh-proxy-20261026
build=$workspace/build-r3d-bootstrap
logs=$workspace/results/r3d-iqr/implementation/logs
mkdir -p "$build" "$logs"
binary=$build/r3dh_moe_w4nl_check
manifest=$logs/r3dh_moe_w4nl-down.manifest
build_log=$logs/r3dh_moe_w4nl-build.log
docker_base=(docker run --rm --user 1001:1001 -v "$workspace:$workspace" -w "$repo")
compiler=("${docker_base[@]}" --entrypoint /opt/rocm/llvm/bin/clang++ "$image")
runner=("${docker_base[@]}" --entrypoint "$binary" "$image")
flags=(-std=c++17 -O2 -fPIC -ffp-contract=off -fno-fast-math -Iabi -Ilibr3d)
"${compiler[@]}" --version > "$build_log" 2>&1
printf 'W4NL_BUILD_FLAGS -std=c++17 -O2 -fPIC -ffp-contract=off -fno-fast-math --offload-arch=gfx1100\n' >> "$build_log"
"${compiler[@]}" "${flags[@]}" -c libr3d/r3dh_moe_w4nl_layout.cpp \
    -o "$build/r3dh_moe_w4nl_layout.o" >> "$build_log" 2>&1
"${compiler[@]}" "${flags[@]}" --offload-arch=gfx1100 -c libr3d/r3dh_moe_w4nl.hip \
    -o "$build/r3dh_moe_w4nl.o" >> "$build_log" 2>&1
"${compiler[@]}" "${flags[@]}" --offload-arch=gfx1100 libr3d/r3dh_moe_w4nl_check.hip \
    "$build/r3dh_moe_w4nl_layout.o" "$build/r3dh_moe_w4nl.o" -o "$binary" >> "$build_log" 2>&1
if "${compiler[@]}" -std=c++17 -ffast-math -fsyntax-only -x c++ -Iabi -Ilibr3d \
    libr3d/r3dh_moe_w4nl.h > "$logs/r3dh_moe_w4nl-fastmath-reject.log" 2>&1; then
    printf 'W4NL_FAIL fast math was accepted\n' >&2
    exit 1
fi
grep -q 'W4NL64 requires' "$logs/r3dh_moe_w4nl-fastmath-reject.log"
if "${compiler[@]}" -std=c++17 -ffinite-math-only -fsyntax-only -x c++ -Iabi -Ilibr3d \
    libr3d/r3dh_moe_w4nl.h > "$logs/r3dh_moe_w4nl-finite-math-reject.log" 2>&1; then
    printf 'W4NL_FAIL finite-only math was accepted\n' >&2
    exit 1
fi
grep -q 'W4NL64 requires' "$logs/r3dh_moe_w4nl-finite-math-reject.log"
"${runner[@]}" --host > "$logs/r3dh_moe_w4nl-host.log" 2>&1
"${runner[@]}" --write-manifest "$model" "$manifest" > "$logs/r3dh_moe_w4nl-manifest-create.log" 2>&1
"${runner[@]}" --manifest-check "$model" "$manifest" > "$logs/r3dh_moe_w4nl-manifest.log" 2>&1
grep 'W4NL_FIXTURE_ABI_PASS' "$logs/r3dh_moe_w4nl-host.log"
grep 'W4NL_REFERENCE_PASS' "$logs/r3dh_moe_w4nl-host.log"
grep 'W4NL_HOST_PASS' "$logs/r3dh_moe_w4nl-host.log"
grep 'W4NL_MANIFEST_PASS' "$logs/r3dh_moe_w4nl-manifest.log"
printf 'W4NL_BUILD_POLICY_PASS fastmath_rejected=1 finite_only_rejected=1\n'
if [[ "$mode" == --host ]]; then
    exit 0
fi
gpu_runner=("${docker_base[@]}" --device /dev/kfd --device /dev/dri --group-add 993 --group-add 44
    --security-opt seccomp=unconfined -e ROCR_VISIBLE_DEVICES=3 -e HIP_VISIBLE_DEVICES=0
    --entrypoint "$binary" "$image")
lock=/home/zihaomu/bigssd/oj_system/var/locks/gpu-renderD131.lock
if { flock -n -E 75 8 && "${gpu_runner[@]}" --gpu; } 8<"$lock" > "$logs/r3dh_moe_w4nl-synthetic.log" 2>&1; then
    grep 'W4NL_PASS' "$logs/r3dh_moe_w4nl-synthetic.log"
else
    status=$?
    printf 'W4NL_GPU_EXIT status=%d (75 means production GPU3 lock busy)\n' "$status" >&2
    exit "$status"
fi
if { flock -n -E 75 8 && "${gpu_runner[@]}" --model "$model" "$manifest"; } 8<"$lock" > "$logs/r3dh_moe_w4nl-model.log" 2>&1; then
    grep 'W4NL_PASS' "$logs/r3dh_moe_w4nl-model.log"
else
    status=$?
    printf 'W4NL_MODEL_EXIT status=%d (75 means production GPU3 lock busy)\n' "$status" >&2
    exit "$status"
fi
printf 'W4NL_ALL_PASS logs=%s\n' "$logs"
