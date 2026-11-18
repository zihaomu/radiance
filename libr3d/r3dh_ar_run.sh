#!/usr/bin/env bash
set -euo pipefail
source_dir=$(dirname "$(realpath "${BASH_SOURCE[0]}")")
repository=$(dirname "$source_dir")
workspace=$(dirname "$repository")
image=freetoken:rocm10.1.0-transformers518-final-gh-proxy-20261026
binary_dir="$workspace/build-r3d-bootstrap"
log_dir="$workspace/results/r3d-iqr/implementation/logs"
mkdir -p "$binary_dir" "$log_dir"
mode=${1:-}
if [[ "$mode" == build ]]; then
    docker run --rm --pull never --network none --user "$(id -u):$(id -g)" -v "$workspace:$workspace" -w "$repository" --entrypoint /opt/rocm/llvm/bin/clang++ "$image" -std=c++20 -O2 -fno-fast-math -ffp-contract=off -x hip --offload-arch=gfx1100 -Iabi -Ilibr3d libr3d/r3dh_ar_check.hip libr3d/r3dh_ar_entry.hip libr3d/r3dh_ar_oneshot_2rank_exact.hip -lamdhip64 -pthread -o "$binary_dir/r3dh_ar_check"
    docker run --rm --pull never --network none --user "$(id -u):$(id -g)" -v "$workspace:$workspace" --entrypoint /opt/rocm/llvm/bin/clang++ "$image" -std=c++20 -O2 -x hip --offload-arch=gfx1100 "$source_dir/r3dh_ar_visibility_check.hip" -lamdhip64 -o "$binary_dir/r3dh_ar_visibility_check"
    exit 0
fi
if [[ "$mode" == check ]]; then
    override=${2:-1}
    command=("$binary_dir/r3dh_ar_check" --gpu)
    label=check
elif [[ "$mode" == repro ]]; then
    kind=${2:-fine}
    if [[ "$kind" != fine && "$kind" != coarse && "$kind" != host ]]; then exit 2; fi
    override=${3:-unset}
    command=("$binary_dir/r3dh_ar_visibility_check" "$kind")
    label="repro-$kind"
else
    printf "usage: bash %s build|check [1|unset]|repro [fine|coarse|host] [unset|1]\n" "$0" >&2
    exit 2
fi
if [[ "$override" != unset && "$override" != 1 ]]; then exit 2; fi
exec 8< /home/zihaomu/bigssd/oj_system/var/locks/gpu-renderD130.lock
exec 9< /home/zihaomu/bigssd/oj_system/var/locks/gpu-renderD131.lock
flock -n 8 || { printf "GPU2 production lock busy\n" >&2; exit 3; }
flock -n 9 || { printf "GPU3 production lock busy\n" >&2; exit 3; }
name="r3dh-ar-run-$label-$$"
cleanup() { docker rm -f "$name" >/dev/null 2>&1 || true; }
trap cleanup EXIT
trap "exit 130" INT
trap "exit 143" TERM
environment=()
if [[ "$override" == 1 ]]; then environment=(-e HSA_FORCE_FINE_GRAIN_PCIE=1); fi
log="$log_dir/r3dh_ar_run-$label-$override.log"
if timeout -k 2s 43s docker run --rm --pull never --network none --name "$name" --user "$(id -u):$(id -g)" --device /dev/kfd --device /dev/dri --group-add 993 --group-add 44 --security-opt seccomp=unconfined -e ROCR_VISIBLE_DEVICES=2,3 -e HIP_VISIBLE_DEVICES=0,1 "${environment[@]}" -v "$workspace:$workspace" --entrypoint /usr/bin/timeout "$image" -s KILL 40s "${command[@]}" > "$log" 2>&1; then
    result=0
else
    result=$?
fi
cat "$log"
printf "RUN_EXIT=%s\n" "$result"
exit "$result"
