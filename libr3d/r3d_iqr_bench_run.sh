#!/usr/bin/env bash
set -euo pipefail
WS=/home/zihaomu/bigssd2/radiance_workspace
REPO="$WS/libr3d-iqr-dev"
IMAGE=freetoken:rocm10.1.0-transformers518-final-gh-proxy-20261026
OUT="${1:?provide a new absolute results/bench directory}"
[[ "$OUT" == "$WS/results/r3d-iqr/implementation/bench-"* ]]
[[ ! -e "$OUT" ]]
mkdir -p "$OUT"
exec > >(tee "$OUT/run.log") 2>&1
trap 'status=$?; jq -n --argjson exit_code "$status" --arg finished "$(date -Is)" "{exit_code:\$exit_code,finished:\$finished}" > "$OUT/exit.json"' EXIT
date -Is
cp "$REPO/libr3d/r3d_iqr_bench_manifest.json" "$OUT/manifest.json"
jq -e '.model_size == 121678372864 and .model_sha256 == "25aff8c950f4b823bc8b77c8184ed48827cb47058ffce08ae2c83b12e021fea0" and .source == "real_row_slice" and .tp2_shard == false' "$OUT/manifest.json" >/dev/null
MODEL=$(jq -r '.model' "$OUT/manifest.json")
[[ $(stat -c %s "$MODEL") == 121678372864 ]]
[[ $(awk 'NR==1 {print $1}' "$WS/models/sha256.txt") == 25aff8c950f4b823bc8b77c8184ed48827cb47058ffce08ae2c83b12e021fea0 ]]
cp "$WS/models/sha256.txt" "$OUT/prior-verified-sha256.txt"
stat -c 'size=%s inode=%i mtime_seconds=%Y ctime_seconds=%Z file=%n' "$MODEL" > "$OUT/model-stat-before.txt"
grep -m1 'blk.0.ssm_inz.weight' "$WS/run/logs/model-tensors.txt" > "$OUT/tensor-evidence.txt"
grep -q '16384x2560.*@53946896384.*codes=u5.*tcq=16.*tcq_code=mad.*tcq_rows=256' "$OUT/tensor-evidence.txt"
sha256sum "$WS/run/logs/model-tensors.txt" "$WS/models/sha256.txt" > "$OUT/evidence-sha256.txt"
while IFS=$'\t' read -r role offset count; do
    digest=$(dd if="$MODEL" iflag=skip_bytes,count_bytes skip="$offset" count="$count" status=none | sha256sum | awk '{print $1}')
    jq -n --arg role "$role" --arg sha256 "$digest" --argjson offset "$offset" --argjson bytes "$count" \
        '{role:$role,sha256:$sha256,offset:$offset,bytes:$bytes}'
done < <(jq -r '.planes | to_entries[] | [.key,.value.offset,.value.bytes] | @tsv' "$OUT/manifest.json") | jq -s '.' > "$OUT/plane-sha256.json"
sha256sum "$REPO/libr3d/r3d_iqr_bench.hip" "$REPO/libr3d/r3d_iqr_bench_host.h" "$REPO/libr3d/r3d_iqr_bench_run.sh" \
    "$REPO/libr3d/r3d_iqr_bench_manifest.json" "$REPO/libr3d/r3d_gemm_iqr.hip" "$REPO/libr3d/r3d_iqr.h" "$REPO/abi/rad_plugin.h" > "$OUT/source-sha256.txt"
git -C "$REPO" rev-parse HEAD > "$OUT/repo-head.txt"
docker image inspect "$IMAGE" --format '{{.Id}}' > "$OUT/sdk-image.txt"
cp "$REPO/libr3d/r3d_iqr_bench.hip" "$REPO/libr3d/r3d_iqr_bench_host.h" "$REPO/libr3d/r3d_gemm_iqr.hip" "$REPO/libr3d/r3d_iqr.h" "$OUT/"
env | grep -E '^REC_' > "$OUT/inherited-rec-env.txt" || true
printf '%s\n' '-std=c++20 -O3 -ffp-contract=off -fno-fast-math --offload-arch=gfx1100 -x hip -Iabi -Ilibr3d' > "$OUT/build-flags.txt"
docker run --rm --user 1001:1001 -v "$WS:$WS" -w "$REPO" --entrypoint /opt/rocm/llvm/bin/clang++ "$IMAGE" \
    -std=c++20 -O3 -ffp-contract=off -fno-fast-math --offload-arch=gfx1100 -x hip -Iabi -Ilibr3d \
    libr3d/r3d_iqr_bench.hip libr3d/r3d_gemm_iqr.hip -o "$OUT/r3d_iqr_bench" > "$OUT/build.log" 2>&1
"$OUT/r3d_iqr_bench" --host > "$OUT/host-tests.json"
"$OUT/r3d_iqr_bench" --model-host "$MODEL" > "$OUT/model-host-tests.json"
docker run --rm --user 1001:1001 -v "$WS:$WS" -w "$REPO" --entrypoint /bin/bash "$IMAGE" -c '
    set -e
    /opt/rocm/llvm/bin/clang++ --version > "$1/compiler.txt"
    /opt/rocm/llvm/bin/clang++ -std=c++20 -O3 -ffp-contract=off -fno-fast-math --offload-arch=gfx1100 --cuda-device-only -x hip -Iabi -Ilibr3d -c libr3d/r3d_gemm_iqr.hip -o "$1/gfx1100.o"
    /opt/rocm/llvm/bin/clang-offload-bundler --type=o --input="$1/gfx1100.o" --list > "$1/offload-targets.txt"
    target=$(head -1 "$1/offload-targets.txt")
    /opt/rocm/llvm/bin/clang-offload-bundler --type=o --input="$1/gfx1100.o" --output="$1/gfx1100.elf" --targets="$target" --unbundle
    /opt/rocm/llvm/bin/llvm-readelf --notes "$1/gfx1100.elf" > "$1/amdgpu-metadata.txt"
    /opt/rocm/llvm/bin/llvm-objdump -d --mcpu=gfx1100 "$1/gfx1100.elf" > "$1/gfx1100.isa"
' bench-isa "$OUT" > "$OUT/isa-build.log" 2>&1 || printf '%s\n' 'ISA_METADATA_EXTRACTION_FAILED; see isa-build.log; no resource guesses' > "$OUT/isa-unavailable.txt"
sha256sum "$OUT/r3d_iqr_bench" > "$OUT/binary-sha256.txt"
{
    flock -n 8 || { printf 'BLOCKED production GPU4 lock busy\n'; exit 75; }
    [[ $(readlink -f /sys/class/drm/renderD132/device) == */0000:83:00.0 ]]
    busy=$(</sys/class/drm/renderD132/device/gpu_busy_percent)
    used=$(</sys/class/drm/renderD132/device/mem_info_vram_used)
    jq -n --argjson busy_percent "$busy" --argjson vram_used_bytes "$used" --arg pci '0000:83:00.0' \
        '{pci:$pci,busy_percent:$busy_percent,vram_used_bytes:$vram_used_bytes,lock:"nonblocking_read_open_production_renderD132"}' > "$OUT/preflight.json"
    [[ "$busy" -eq 0 && "$used" -lt 536870912 ]] || { printf 'BLOCKED GPU4 not idle\n'; exit 76; }
    docker run --rm --name "r3d-iqr-bench-gpu4-$$" --user 1001:1001 --device /dev/kfd --device /dev/dri \
        --group-add 993 --group-add 44 --security-opt seccomp=unconfined -e ROCR_VISIBLE_DEVICES=4 -e HIP_VISIBLE_DEVICES=0 \
        -v "$WS:$WS" -w "$REPO" --entrypoint "$OUT/r3d_iqr_bench" "$IMAGE" "$MODEL" "$OUT" > "$OUT/gpu-run.log" 2>&1
    cat /sys/class/drm/renderD132/device/gpu_busy_percent /sys/class/drm/renderD132/device/mem_info_vram_used > "$OUT/gpu-after.txt"
} 8</home/zihaomu/bigssd/oj_system/var/locks/gpu-renderD132.lock
sha256sum -c "$OUT/source-sha256.txt" > "$OUT/source-stability.txt"
stat -c 'size=%s inode=%i mtime_seconds=%Y ctime_seconds=%Z file=%n' "$MODEL" > "$OUT/model-stat-after.txt"
cmp "$OUT/model-stat-before.txt" "$OUT/model-stat-after.txt"
jq -e '.status == "pass" and .timed_cases == 36 and .timed_batches == 3600' "$OUT/completed.json" >/dev/null
awk -F, 'NR>1 { count++; if ($10<1 || $12<=0 || $13<2000) bad++; key=$2 FS $5 FS $9; cases[key]++; samples[key FS $6]++ } END {for (key in cases) {nc++; if(cases[key]!=100) bad++} for(key in samples) if(samples[key]!=20) bad++; print "batches=" count,"cases=" nc,"invalid=" bad+0; exit(count!=3600 || nc!=36 || bad)}' "$OUT/raw.csv" > "$OUT/raw-validation.txt"
awk -F, 'NR>1 {cases++; outputs+=$4; mismatches+=$5; guards+=$6} END {print "checks="cases,"cpu_output_comparisons="outputs,"mismatches="mismatches,"guard_elements="guards; exit(cases!=36 || mismatches)}' "$OUT/correctness.csv" > "$OUT/correctness-validation.txt"
{
    flock -n 8
    printf 'production_lock_reacquired_after_run=true\n'
} 8</home/zihaomu/bigssd/oj_system/var/locks/gpu-renderD132.lock
cat "$OUT/completed.json" "$OUT/raw-validation.txt" "$OUT/correctness-validation.txt"
date -Is
