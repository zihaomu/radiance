#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
workspace=$(dirname "$root")
build="$workspace/build-r3dh-gdn-chunk-scope"
image=${R3DH_GDN_EXTRA_SDK:-freetoken:rocm10.1.0-transformers518-final-gh-proxy-20261026}
if [[ ${1:-} == --inside ]]; then
    cd "$root"
    mkdir -p "$build"
    cxx=/opt/rocm/llvm/bin/clang++
    flags=(-std=c++20 -O2 -fPIC -fno-fast-math -ffp-contract=off -ffunction-sections -fdata-sections -Iabi -Ilibr3d -Ilibref)
    printf "GDN_CHUNK_BUILD user=%s target=gfx1100 flags=%s\n" "$(id -u)" "${flags[*]}"
    "$cxx" --version
    for unit in strict chunk_strict; do
        "$cxx" "${flags[@]}" --offload-arch=gfx1100 -c "libr3d/r3dh_gdn_extra_${unit}.hip" -o "$build/$unit.o"
    done
    for unit in gdn quant; do
        "$cxx" "${flags[@]}" -c "libref/ref_${unit}.cpp" -o "$build/ref_$unit.o"
    done
    "$cxx" "${flags[@]}" -c libr3d/r3dh_gdn_extra_check_chunk_launch.cpp -o "$build/launch.o"
    "$cxx" "${flags[@]}" --offload-arch=gfx1100 -DR3DH_GDN_CHUNK_CHECK_REGISTRY -c libr3d/r3dh_gdn_extra_check.hip -o "$build/check.o"
    "$cxx" --hip-link -Wl,--gc-sections "$build/check.o" "$build/strict.o" "$build/chunk_strict.o" "$build/launch.o" "$build/ref_gdn.o" "$build/ref_quant.o" -lamdhip64 -o "$build/check"
    sha256sum libr3d/r3dh_gdn_extra_chunk_strict.hip libr3d/r3dh_gdn_extra_check.hip libr3d/r3dh_gdn_extra_check_chunk* libr3d/r3d_model_extra_registry.cpp "$build/check"
    printf "PASS GDN_CHUNK_BUILD native=gfx1100 contract=off private_objects\n"
    exit 0
fi
[[ $# == 0 ]] || { printf "usage: bash %s [--inside]\n" "$0" >&2; exit 2; }
logs="$workspace/results/r3d-iqr/implementation/logs"
mkdir -p "$logs"
stamp=$(date -u +%Y%m%dT%H%M%SZ)
prefix="$logs/r3dh-gdn-chunk-multi-$stamp"
docker run --rm --user 1001:1001 -v "$workspace:$workspace" -w "$root" --entrypoint /bin/bash "$image" "$root/libr3d/r3dh_gdn_extra_check_chunk_run.sh" --inside > "$prefix-build.log" 2>&1
run_locked() {
    flock -n 8 || { printf "GDN_CHUNK_LOCK_BUSY renderD135\n"; return 75; }
    printf "GDN_CHUNK_LOCK_ACQUIRED GPU7 renderD135\n"
    local processes
    processes=$(amd-smi process --gpu 7)
    printf "%s\n" "$processes"
    [[ "$processes" == *"No running processes detected"* ]] || { printf "GDN_CHUNK_GPU7_NOT_IDLE\n"; return 75; }
    for mode in --chunk --chunk-registry; do
        printf "GDN_CHUNK_MODE=%s\n" "$mode"
        timeout --signal=TERM --kill-after=5s 1200s docker run --rm --user 1001:1001 --device /dev/kfd --device /dev/dri --group-add 993 --group-add 44 --security-opt seccomp=unconfined -e ROCR_VISIBLE_DEVICES=7 -e HIP_VISIBLE_DEVICES=0 -v "$workspace:$workspace" -w "$root" --entrypoint "$build/check" "$image" "$mode" || return "$?"
    done
}
status=0
run_locked 8</home/zihaomu/bigssd/oj_system/var/locks/gpu-renderD135.lock > "$prefix-gpu.log" 2>&1 || status=$?
printf "build_log=%s\ngpu_log=%s\n" "$prefix-build.log" "$prefix-gpu.log"
tail -6 "$prefix-gpu.log"
exit "$status"
