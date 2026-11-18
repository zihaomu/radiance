#!/usr/bin/env bash
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
workspace=$(dirname -- "$repo")
mode=${1:---host}
if [[ "$mode" != --host && "$mode" != --all ]]; then printf 'Usage: bash libr3d/r3dh_vision_strict_test.sh --host|--all\n' >&2; exit 2; fi
image=freetoken:rocm10.1.0-transformers518-final-gh-proxy-20261026
build=$workspace/build-r3d-bootstrap
logs=$workspace/results/r3d-iqr/implementation/logs
mkdir -p "$build" "$logs"
binary=$build/r3dh_vision_strict_check
base=(docker run --rm --user 1001:1001 -v "$workspace:$workspace" -w "$repo")
compiler=("${base[@]}" --entrypoint /opt/rocm/llvm/bin/clang++ "$image")
flags=(-std=c++17 -O2 -fPIC -ffp-contract=off -fno-fast-math --offload-arch=gfx1100 -Iabi -Ilibr3d)
"${compiler[@]}" --version > "$logs/r3dh_vision_strict-build.log" 2>&1
printf 'VISION_FLAGS %s\n' "${flags[*]}" >> "$logs/r3dh_vision_strict-build.log"
"${compiler[@]}" "${flags[@]}" libr3d/r3dh_vision_strict.hip libr3d/r3dh_vision_strict_shapes.cpp \
    libr3d/r3dh_vision_strict_registry.cpp libr3d/r3dh_vision_strict_check.hip -o "$binary" >> "$logs/r3dh_vision_strict-build.log" 2>&1
for forbidden in -ffast-math -ffinite-math-only; do
    if "${compiler[@]}" -std=c++17 "$forbidden" -x c++ -fsyntax-only -Iabi -Ilibr3d libr3d/r3dh_vision_strict.h > "$logs/r3dh_vision_strict-reject${forbidden}.log" 2>&1; then
        printf 'VISION_FAIL accepted %s\n' "$forbidden" >&2; exit 1
    fi
    grep -q 'requires strict floating point' "$logs/r3dh_vision_strict-reject${forbidden}.log"
done
"${base[@]}" --entrypoint "$binary" "$image" --host > "$logs/r3dh_vision_strict-host.log" 2>&1
grep VISION_HOST_PASS "$logs/r3dh_vision_strict-host.log"
if [[ "$mode" == --host ]]; then exit 0; fi
reference=$workspace/build-r3d-gfx1100/radiance_home/kernels/libref.so
gpu=("${base[@]}" --device /dev/kfd --device /dev/dri --group-add 993 --group-add 44 --security-opt seccomp=unconfined
    -e ROCR_VISIBLE_DEVICES=5 -e HIP_VISIBLE_DEVICES=0 -e "VISION_REF_LIBRARY=$reference" -e OMP_NUM_THREADS=4 --entrypoint "$binary" "$image")
lock=/home/zihaomu/bigssd/oj_system/var/locks/gpu-renderD133.lock
run_gpu() {
    flock -n -E 75 8 || return $?
    if [[ $(< /sys/class/drm/renderD133/device/gpu_busy_percent) -ne 0 || $(< /sys/class/drm/renderD133/device/mem_info_vram_used) -ge 1073741824 ]]; then
        printf 'VISION_GPU5_NOT_IDLE\n' >&2; return 75
    fi
    "${gpu[@]}" --gpu "$@"
}
run_gpu 8<"$lock" > "$logs/r3dh_vision_strict-synthetic.log" 2>&1
grep -E 'VISION_(GPU|LIBREF)_PASS' "$logs/r3dh_vision_strict-synthetic.log"
run_gpu "$workspace/models/qwen3.8-next-flash-iqr-v4.rad" 8<"$lock" > "$logs/r3dh_vision_strict-model.log" 2>&1
grep -E 'VISION_(GPU|LIBREF)_PASS' "$logs/r3dh_vision_strict-model.log"
printf 'VISION_ALL_PASS logs=%s\n' "$logs"
