#!/usr/bin/env bash
set -euo pipefail
ws=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ws"
out=${1:-results/r3d-iqr/implementation/quick-operators-$(date +%Y%m%d-%H%M%S)}
mkdir -p "$out"
[[ ! -e "$out/summary.jsonl" ]] || { printf 'Refusing to overwrite %s\n' "$out/summary.jsonl"; exit 1; }
exec 8</home/zihaomu/bigssd/oj_system/var/locks/gpu-renderD132.lock
exec 9</home/zihaomu/bigssd/oj_system/var/locks/gpu-renderD133.lock
flock -n -E 75 8
flock -n -E 75 9
if env | grep -E '^REC_'; then printf 'Unset experiment REC_ variables before this run\n'; exit 1; fi
name=r3d-quick-operators-$$
tag=initial
cleanup() {
    docker logs "$name" > "$out/$tag-server.log" 2>&1 || true
    docker stop -t 30 "$name" >/dev/null 2>&1 || true
}
trap cleanup EXIT
sha256sum build-r3d-gfx1100/radiance_home/kernels/libr3d.so > "$out/library.sha256"
read -r -a modes <<< "${QUICK_OPERATOR_MODES:-mtp0 mtp3}"
read -r -a variants <<< "${QUICK_OPERATOR_VARIANTS:-base:0:0:0 exact:1:1:0}"
for mode in "${modes[@]}"; do
    [[ "$mode" == mtp0 || "$mode" == mtp3 ]] || exit 2
    for variant in "${variants[@]}"; do
        IFS=: read -r label iqr logits hc prefetch hc_tile dense iqr_dense norm inject extra <<< "$variant"
        prefetch=${prefetch:-0}
        hc_tile=${hc_tile:-0}
        dense=${dense:-0}
        iqr_dense=${iqr_dense:-0}
        norm=${norm:-0}
        inject=${inject:-0}
        [[ "$label" =~ ^[a-zA-Z0-9_-]+$ && "$iqr" =~ ^[01]$ && "$logits" =~ ^[01]$ && "$hc" =~ ^[01]$ \
            && "$prefetch" =~ ^[01]$ && "$hc_tile" =~ ^[01]$ && "$dense" =~ ^[01]$ \
            && "$iqr_dense" =~ ^[01]$ && "$norm" =~ ^[01]$ && "$inject" =~ ^[01]$ && -z "$extra" ]] || exit 2
        tag=$mode-$label
        amd-smi process --gpu 4 5 --json | jq -e 'all(.[].process_list[]; .process_info == "No running processes detected")' >/dev/null
        printf 'Serving options: mode=%s IQR_DOT4=%s LOGITS_STAGED=%s HC_WMMA=%s IQR_PREFETCH=%s HC_EXACT_TILE=%s I8_DOT4=%s IQR_DENSE_WAVE=%s HC_NORM_PARALLEL=%s HC_INJECT_PARALLEL=%s profile=off GPU=4,5\n' \
            "$mode" "$iqr" "$logits" "$hc" "$prefetch" "$hc_tile" "$dense" "$iqr_dense" "$norm" "$inject"
        docker run -d --rm --name "$name" --user 1001:1001 --device /dev/kfd --device /dev/dri \
            --group-add 993 --group-add 44 --security-opt seccomp=unconfined --ulimit memlock=-1 --ipc=host \
            -e ROCR_VISIBLE_DEVICES=4,5 -e HIP_VISIBLE_DEVICES=0,1 -e HSA_FORCE_FINE_GRAIN_PCIE=1 \
            -e RAD_IQR_DOT4="$iqr" -e RAD_LOGITS_STAGED="$logits" -e RAD_HC_WMMA="$hc" \
            -e RAD_IQR_PREFETCH="$prefetch" -e RAD_HC_EXACT_TILE="$hc_tile" -e RAD_I8_DOT4="$dense" \
            -e RAD_IQR_DENSE_WAVE="$iqr_dense" \
            -e RAD_HC_NORM_PARALLEL="$norm" \
            -e RAD_HC_INJECT_PARALLEL="$inject" \
            -v "$ws:$ws" -w "$ws" -p 127.0.0.1:18103:8000 \
            --entrypoint "$ws/build-r3d-gfx1100/bin/radiance" radiance-r3d-dev:rocm10.1-ffmpeg \
            --config "$ws/run/bench-tp2-$mode.yaml" > "$out/$tag-container.txt"
        curl -sS --fail --retry 150 --retry-delay 2 --retry-connrefused --retry-all-errors --max-time 3 \
            http://127.0.0.1:18103/v1/models -o /dev/null 2> "$out/$tag-ready.log"
        curl -sS --fail --max-time 300 -H 'Content-Type: application/json' \
            --data-binary @run/native-text-192-request.json http://127.0.0.1:18103/v1/chat/completions \
            -o "$out/$tag-warmup.json"
        for workload in short arithmetic decode128; do
            case "$workload" in
                short) request=native-text-192-request.json ;;
                arithmetic) request=native-long-512-request.json ;;
                decode128) request=bench-decode-128-request.json ;;
            esac
            response=$out/$tag-$workload.json
            curl -sS --fail --max-time 300 -H 'Content-Type: application/json' \
                --data-binary "@run/$request" http://127.0.0.1:18103/v1/chat/completions -o "$response"
            jq -e '.timings.predicted_per_second > 0 and (.choices | length == 1)' "$response" >/dev/null
            jq -c --arg tag "$tag" --arg workload "$workload" \
                '{tag:$tag,workload:$workload,prompt_n:.timings.prompt_n,generated_n:.timings.predicted_n,decode_tps:.timings.predicted_per_second,answer:.choices[0].message.content,finish:.choices[0].finish_reason}' \
                "$response" | tee -a "$out/summary.jsonl"
        done
        cleanup
    done
done
trap - EXIT
printf 'Completed quick serving A/B: %s\n' "$out"
