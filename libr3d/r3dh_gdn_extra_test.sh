#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
workspace=$(dirname "$root")
image=${R3DH_GDN_EXTRA_SDK:-freetoken:rocm10.1.0-transformers518-final-gh-proxy-20261026}
log_dir=${R3DH_GDN_EXTRA_LOG_DIR:-$workspace/results/r3d-iqr/implementation/logs}
stamp=$(date -u +%Y%m%dT%H%M%SZ)
build_log="$log_dir/r3dh-gdn-extra-$stamp-build.log"
gpu_log="$log_dir/r3dh-gdn-extra-$stamp-gpu.log"
mkdir -p "$log_dir"
bash "$root/libr3d/r3dh_gdn_extra_build.sh" > "$build_log" 2>&1
run_locked() {
    if ! flock -n 8; then
        printf 'GDN_EXTRA_LOCK_BUSY renderD130 GPU2\n'
        return 75
    fi
    printf 'GDN_EXTRA_LOCK_ACQUIRED renderD130 GPU2\n'
    for repeat in 1 2; do
        printf 'GDN_EXTRA_REPEAT=%s\n' "$repeat"
        timeout --signal=TERM --kill-after=5s 240s docker run --rm \
            --user 1001:1001 --device /dev/kfd --device /dev/dri \
            --group-add 993 --group-add 44 --security-opt seccomp=unconfined \
            -e ROCR_VISIBLE_DEVICES=2 -e HIP_VISIBLE_DEVICES=0 \
            -v "$workspace:$workspace" -w "$root" \
            --entrypoint "$workspace/build-r3d-bootstrap/r3dh_gdn_extra_check" \
            "$image" || return "$?"
    done
}
status=0
run_locked 8</home/zihaomu/bigssd/oj_system/var/locks/gpu-renderD130.lock > "$gpu_log" 2>&1 || status=$?
printf 'build_log=%s\ngpu_log=%s\n' "$build_log" "$gpu_log"
if [[ "$status" == 0 ]]; then
    awk '/^PASS decode /{decode++} /^PASS anchor_contract /{anchor+=3} /^PASS chunk_contract /{chunk+=3} /^PASS prefill /{prefill++} /^PASS GDN_EXTRA /{runs++} END{printf "PASS repeats=%d decode_steps=%d anchor_assertions=%d prefill_pipelines=%d chunk_assertions=%d tolerance=0\n",runs,decode,anchor,prefill,chunk}' "$gpu_log"
else
    tail -8 "$gpu_log"
fi
exit "$status"
