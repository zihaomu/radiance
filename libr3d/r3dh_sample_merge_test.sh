#!/usr/bin/env bash
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
workspace=$(dirname -- "$repo")
mode=${1:---host}
if [[ $# -gt 1 || ( "$mode" != --host && "$mode" != --all && "$mode" != --sanitize ) ]]; then
    printf 'Usage: bash libr3d/r3dh_sample_merge_test.sh --host|--all|--sanitize\n' >&2
    exit 2
fi
image=freetoken:rocm10.1.0-transformers518-final-gh-proxy-20261026
build=$workspace/build-r3d-bootstrap
logs=$workspace/results/r3d-iqr/implementation/logs
mkdir -p "$build" "$logs"
binary=$build/r3dh_sample_merge_check
base=(docker run --rm --user 1001:1001 -v "$workspace:$workspace" -w "$repo")
compiler=("${base[@]}" --entrypoint /opt/rocm/llvm/bin/clang++ "$image")
flags=(-std=c++17 -O2 -Wall -Wextra -fno-fast-math -ffp-contract=off -Iabi -Ilibr3d -Ilibref)
extra=()
suffix=
if [[ "$mode" == --sanitize ]]; then
    extra=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer --offload-host-only)
    suffix=-sanitized
    binary=$binary$suffix
fi
log=$logs/r3dh_sample_merge-build$suffix.log
"${compiler[@]}" --version > "$log" 2>&1
printf 'BUILD uid=1001 same_path=%s image=%s flags=%s %s\n' "$workspace" "$image" "${flags[*]}" "${extra[*]}" >> "$log"
host_extra=()
if [[ "$mode" == --sanitize ]]; then host_extra=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer); fi
"${compiler[@]}" "${flags[@]}" "${host_extra[@]}" -c libref/ref_sample.cpp \
    -o "$build/r3dh_sample_merge_ref$suffix.o" >> "$log" 2>&1
"${compiler[@]}" "${flags[@]}" "${extra[@]}" --offload-arch=gfx1100 \
    libr3d/r3dh_sample_merge.hip libr3d/r3dh_sample_merge_registry.cpp \
    libr3d/r3dh_sample_merge_check.hip "$build/r3dh_sample_merge_ref$suffix.o" \
    -o "$binary" >> "$log" 2>&1
"${base[@]}" -e ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 -e UBSAN_OPTIONS=halt_on_error=1 \
    --entrypoint "$binary" "$image" --host > "$logs/r3dh_sample_merge-host$suffix.log" 2>&1
grep 'SAMPLE_MERGE_PASS' "$logs/r3dh_sample_merge-host$suffix.log"
if [[ "$mode" != --all ]]; then exit 0; fi
for forbidden in -ffast-math -ffinite-math-only; do
    if "${compiler[@]}" -std=c++17 "$forbidden" --offload-arch=gfx1100 -Iabi -Ilibr3d \
        -fsyntax-only libr3d/r3dh_sample_merge.hip > "$logs/r3dh_sample_merge-reject$forbidden.log" 2>&1; then
        printf 'FAIL accepted %s\n' "$forbidden" >&2; exit 1
    fi
    grep -q 'sample_merge requires strict NaN and infinity semantics' "$logs/r3dh_sample_merge-reject$forbidden.log"
done
"${compiler[@]}" "${flags[@]}" --offload-arch=gfx1100 --offload-device-only -S \
    libr3d/r3dh_sample_merge.hip -o "$build/r3dh_sample_merge-gfx1100.s" >> "$log" 2>&1
grep -E 'amdhsa_(group_segment_fixed_size|private_segment_fixed_size|next_free_vgpr|next_free_sgpr)' \
    "$build/r3dh_sample_merge-gfx1100.s" > "$logs/r3dh_sample_merge-resources.log"
run_gpu() {
    local gpu=$1 render=$2
    flock -n -E 75 8 || return $?
    local busy=/sys/class/drm/renderD$render/device/gpu_busy_percent
    local memory=/sys/class/drm/renderD$render/device/mem_info_vram_used
    if [[ $(< "$busy") -ne 0 || $(< "$memory") -ge 1073741824 ]]; then return 75; fi
    "${base[@]}" --name "r3dh-sample-merge-gpu$gpu" --device /dev/kfd --device /dev/dri \
        --group-add 993 --group-add 44 --security-opt seccomp=unconfined \
        -e "ROCR_VISIBLE_DEVICES=$gpu" -e HIP_VISIBLE_DEVICES=0 --entrypoint "$binary" \
        "$image" --gpu > "$logs/r3dh_sample_merge-gpu$gpu.log" 2>&1
}
set +e
run_gpu 5 133 8</home/zihaomu/bigssd/oj_system/var/locks/gpu-renderD133.lock
status=$?
set -e
chosen=5
if [[ $status -eq 75 ]]; then
    set +e
    run_gpu 4 132 8</home/zihaomu/bigssd/oj_system/var/locks/gpu-renderD132.lock
    status=$?
    set -e
    chosen=4
fi
if [[ $status -ne 0 ]]; then printf 'SAMPLE_MERGE_GPU_EXIT=%s\n' "$status" >&2; exit "$status"; fi
grep -E 'DEVICE|SAMPLE_MERGE_PASS' "$logs/r3dh_sample_merge-gpu$chosen.log"
sha256sum "$binary" "$repo"/libr3d/r3dh_sample_merge{.h,.hip,_registry.cpp,_check.hip,_test.sh} \
    > "$logs/r3dh_sample_merge-sha256.log"
printf 'SAMPLE_MERGE_ALL_PASS gpu=%s logs=%s\n' "$chosen" "$logs"
