#!/usr/bin/env bash
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
workspace=$(dirname "$repo")
mode=${1:-all}
mode=${mode#--}
case "$mode" in build|host|gpu|all) ;; *) printf 'usage: %s build|host|gpu|all\n' "$0" >&2; exit 2 ;; esac
build=$workspace/build-r3d-bootstrap
logs=$workspace/results/r3d-iqr/implementation/logs
mkdir -p "$build" "$logs"
binary=$build/r3dh_attn_qsa_check
shared=$build/r3dh_attn_qsa.so
image=freetoken:rocm10.1.0-transformers518-final-gh-proxy-20261026
docker_base=(docker run --rm --user 1001:1001 -v "$workspace:$workspace" -w "$repo")
compiler=("${docker_base[@]}" --entrypoint /opt/rocm/llvm/bin/clang++ "$image")
runner=("${docker_base[@]}" --entrypoint "$binary" "$image")
flags=(-std=c++17 -O2 -fPIC -ffp-contract=off -fno-fast-math -Iabi -Ilibr3d -Ilibref)
sources=(r3dh_attn_qsa_device r3dh_attn_qsa_rope r3dh_attn_qsa_state)
objects=()
for source in "${sources[@]}"; do objects+=("$build/$source.o"); done
if [[ "$mode" == build || "$mode" == all ]]; then
    "${compiler[@]}" --version > "$logs/r3dh_attn_qsa-build.log" 2>&1
    printf 'QSA_BUILD_FLAGS %s --offload-arch=gfx1100\n' "${flags[*]}" >> "$logs/r3dh_attn_qsa-build.log"
    for source in ref_attn ref_qsa; do
        "${compiler[@]}" "${flags[@]}" -c "libref/$source.cpp" -o "$build/r3dh_attn_qsa_$source.o" \
            >> "$logs/r3dh_attn_qsa-build.log" 2>&1
    done
    for source in "${sources[@]}"; do
        "${compiler[@]}" "${flags[@]}" --offload-arch=gfx1100 -c "libr3d/$source.hip" -o "$build/$source.o" \
            >> "$logs/r3dh_attn_qsa-build.log" 2>&1
    done
    "${compiler[@]}" "${flags[@]}" --offload-arch=gfx1100 --hip-link -shared "${objects[@]}" -lamdhip64 -o "$shared" \
        >> "$logs/r3dh_attn_qsa-build.log" 2>&1
    "${compiler[@]}" "${flags[@]}" --offload-arch=gfx1100 libr3d/r3dh_attn_qsa_check.hip "${objects[@]}" \
        "$build/r3dh_attn_qsa_ref_attn.o" "$build/r3dh_attn_qsa_ref_qsa.o" -o "$binary" \
        >> "$logs/r3dh_attn_qsa-build.log" 2>&1
    for policy in fast-math finite-math-only; do
        if "${compiler[@]}" -std=c++17 "-f$policy" -fsyntax-only -x c++ -Iabi -Ilibr3d libr3d/r3dh_attn_qsa.h \
            > "$logs/r3dh_attn_qsa-$policy-reject.log" 2>&1; then
            printf 'QSA_FAIL accepted %s\n' "$policy" >&2
            exit 1
        fi
        grep -q 'requires strict floating point' "$logs/r3dh_attn_qsa-$policy-reject.log"
    done
    if nm -D "$shared" | grep -E 'ref_attn|ref_qsa|cpu_prepare'; then
        printf 'QSA_FAIL production library links a CPU oracle\n' >&2
        exit 1
    fi
    nm -D --defined-only "$shared" | grep ' T r3dh_attn_qsa_' > "$logs/r3dh_attn_qsa-exports.log"
    [[ $(wc -l < "$logs/r3dh_attn_qsa-exports.log") == 8 ]]
    if "${runner[@]}" --invalid > "$logs/r3dh_attn_qsa-cli-reject.log" 2>&1; then
        printf 'QSA_FAIL accepted invalid checker mode\n' >&2
        exit 1
    else
        [[ $? == 2 ]]
    fi
    printf 'QSA_BUILD_PASS cxx=17 exports=8 cpu_oracles_linked=0 fastmath_rejected=1 finite_only_rejected=1\n'
fi
if [[ "$mode" != gpu ]]; then
    "${runner[@]}" --host > "$logs/r3dh_attn_qsa-host.log" 2>&1
    tail -n 2 "$logs/r3dh_attn_qsa-host.log"
fi
if [[ "$mode" == gpu || "$mode" == all ]]; then
    lock=/home/zihaomu/bigssd/oj_system/var/locks/gpu-renderD131.lock
    if { flock -n -E 75 8 && "${docker_base[@]}" --device /dev/kfd --device /dev/dri --group-add 993 --group-add 44 \
        --security-opt seccomp=unconfined -e ROCR_VISIBLE_DEVICES=3 -e HIP_VISIBLE_DEVICES=0 \
        --entrypoint "$binary" "$image" --gpu; } 8<"$lock" > "$logs/r3dh_attn_qsa-gpu.log" 2>&1; then
        tail -n 3 "$logs/r3dh_attn_qsa-gpu.log"
    else
        status=$?
        printf 'QSA_GPU_EXIT=%d (75 means production GPU3 lock busy)\n' "$status" >&2
        exit "$status"
    fi
fi
printf 'QSA_TEST_PASS mode=%s logs=%s\n' "$mode" "$logs"
