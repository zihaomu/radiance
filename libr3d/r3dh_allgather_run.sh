#!/usr/bin/env bash
set -euo pipefail
source_dir=$(dirname "$(realpath "${BASH_SOURCE[0]}")")
repository=$(dirname "$source_dir")
workspace=$(dirname "$repository")
image=radiance-r3d-dev:rocm10.1-ffmpeg
binary_dir="$workspace/build-r3d-bootstrap"
log_dir="$workspace/results/r3d-iqr/implementation/logs"
compiler=/opt/rocm/llvm/bin/clang++
mode=${1:---all}
if [[ $# -gt 0 ]]; then shift; fi
physical_gpus=6,7
authorized_bdfs=
custom_gpus=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --physical-gpus) [[ $# -ge 2 ]] || exit 2; physical_gpus=$2; custom_gpus=1; shift 2 ;;
        --authorize-pci-bdfs) [[ $# -ge 2 ]] || exit 2; authorized_bdfs=$2; shift 2 ;;
        *) printf 'unknown option: %s\n' "$1" >&2; exit 2 ;;
    esac
done
if [[ ! "$physical_gpus" =~ ^[0-7],[0-7]$ || "${physical_gpus:0:1}" == "${physical_gpus:2:1}" ]]; then exit 2; fi
if [[ "$custom_gpus" == 1 && -z "$authorized_bdfs" || "$custom_gpus" == 0 && -n "$authorized_bdfs" ]]; then
    printf 'custom GPU selection requires both --physical-gpus and --authorize-pci-bdfs\n' >&2
    exit 2
fi
IFS=, read -r -a ordinals <<< "$physical_gpus"
renders=("renderD$((128+ordinals[0]))" "renderD$((128+ordinals[1]))")
mkdir -p "$binary_dir" "$log_dir"
if [[ "$mode" == --inside-build ]]; then
    cd "$repository"
    "$compiler" -std=c++20 -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer \
        -DR3DH_ALLGATHER_HOST_ONLY -Iabi -Ilibr3d libr3d/r3dh_allgather_check.cpp \
        libr3d/r3dh_allgather_host.cpp -o "$binary_dir/r3dh_allgather_host_check"
    ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
        "$binary_dir/r3dh_allgather_host_check" --host > "$log_dir/r3dh_allgather-host-asan.log" 2>&1
    "$compiler" -std=c++20 -O2 -fPIC -shared -x hip --offload-arch=gfx1100 -Iabi -Ilibr3d \
        libr3d/r3dh_allgather.hip libr3d/r3dh_allgather_host.cpp -lamdhip64 -pthread \
        -o "$binary_dir/r3dh_allgather.so"
    "$compiler" -std=c++20 -O2 -x hip --offload-arch=gfx1100 -Iabi -Ilibr3d \
        libr3d/r3dh_allgather_check.cpp libr3d/r3dh_allgather.hip libr3d/r3dh_allgather_host.cpp \
        -lamdhip64 -pthread -o "$binary_dir/r3dh_allgather_check"
    printf 'ALLGATHER_BUILD_PASS host_asan_ubsan=1 gfx1100=1\n'
    exit 0
fi
if [[ "$mode" == --build || "$mode" == --all ]]; then
    docker run --rm --pull never --network none --user 1001:1001 \
        -v "$workspace:$workspace" --entrypoint /bin/bash "$image" \
        "$source_dir/r3dh_allgather_run.sh" --inside-build > "$log_dir/r3dh_allgather-build.log" 2>&1
    tail -n 4 "$log_dir/r3dh_allgather-build.log"
    cat "$log_dir/r3dh_allgather-host-asan.log"
    if [[ "$mode" == --build ]]; then exit 0; fi
fi
if [[ "$mode" != --gpu && "$mode" != --all && "$mode" != --frozen && "$mode" != --aql ]]; then
    printf 'usage: bash %s [--build|--gpu|--frozen|--aql|--all] [--physical-gpus X,Y --authorize-pci-bdfs BDF,BDF]\n' "$0" >&2
    exit 2
fi
exec 8< "/home/zihaomu/bigssd/oj_system/var/locks/gpu-${renders[0]}.lock"
exec 9< "/home/zihaomu/bigssd/oj_system/var/locks/gpu-${renders[1]}.lock"
flock -n 8 || { printf '%s production lock busy\n' "${renders[0]}" >&2; exit 75; }
flock -n 9 || { printf '%s production lock busy\n' "${renders[1]}" >&2; exit 75; }
actual_bdfs="$(basename "$(readlink -f "/sys/class/drm/${renders[0]}/device")"),$(basename "$(readlink -f "/sys/class/drm/${renders[1]}/device")")"
if [[ "$custom_gpus" == 1 && "$authorized_bdfs" != "$actual_bdfs" ]]; then
    printf 'PCI authorization mismatch expected=%s actual=%s\n' "$authorized_bdfs" "$actual_bdfs" >&2
    exit 2
fi
printf 'ALLGATHER_AUTH physical_gpus=%s pci=%s locks=%s,%s\n' "$physical_gpus" "$actual_bdfs" "${renders[0]}" "${renders[1]}"
for render in "${renders[@]}"; do
    read -r busy < "/sys/class/drm/$render/device/gpu_busy_percent"
    read -r used < "/sys/class/drm/$render/device/mem_info_vram_used"
    printf 'ALLGATHER_IDLE_GATE device=%s busy=%s vram_bytes=%s\n' "$render" "$busy" "$used"
    if [[ "$busy" != 0 || "$used" -gt 67108864 ]]; then
        printf '%s is not idle\n' "$render" >&2
        exit 75
    fi
done
test_mode=--gpu
if [[ "$mode" == --frozen ]]; then test_mode=--frozen; fi
test_binary="$binary_dir/r3dh_allgather_check"
test_arguments=("$test_mode")
if [[ "$mode" == --aql ]]; then
    test_mode=--aql
    test_binary="$workspace/build-r3d-gfx1100/tests/r3d_mtp_ple_device_test"
    test_arguments=(r3d_allgather_)
fi
gpu_log="$log_dir/r3dh_allgather-${test_mode#--}.log"
name="r3dh-allgather-check-$$"
cleanup() { docker rm -f "$name" >/dev/null 2>&1 || true; }
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
if timeout -k 3s 165s docker run --rm --pull never --network none --name "$name" \
    --user 1001:1001 --device /dev/kfd --device /dev/dri --group-add 993 --group-add 44 \
    --security-opt seccomp=unconfined -e ROCR_VISIBLE_DEVICES="$physical_gpus" -e HIP_VISIBLE_DEVICES=0,1 \
    -e HSA_FORCE_FINE_GRAIN_PCIE=1 -v "$workspace:$workspace" \
    -e RADIANCE_HOME="$workspace/build-r3d-gfx1100/radiance_home" \
    -e RAD_R3D_REQUIRE_DEVICE=1 -e RAD_R3D_REQUIRE_TP2=1 -e RAD_TEST_CPU_BUDGET=0 \
    --entrypoint /usr/bin/timeout "$image" -s KILL 150s "$test_binary" "${test_arguments[@]}" \
    > "$gpu_log" 2>&1; then
    tail -n 6 "$gpu_log"
    printf 'ALLGATHER_RUN_PASS physical_gpus=%s production_locks=%s,%s\n' "$physical_gpus" "${renders[0]}" "${renders[1]}"
else
    result=$?
    tail -n 25 "$gpu_log"
    printf 'ALLGATHER_RUN_FAIL exit=%s\n' "$result" >&2
    exit "$result"
fi
