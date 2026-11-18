#!/usr/bin/env bash
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
workspace=$(dirname "$repo")
mode=${1:---all}
mode=${mode#--}
case "$mode" in build|host|gpu|all) ;; *) printf 'usage: bash %s --build|--host|--gpu|--all\n' "$0" >&2; exit 2 ;; esac
build=$workspace/build-r3d-bootstrap
logs=$workspace/results/r3d-iqr/implementation/logs
mkdir -p "$build" "$logs"
binary=$build/r3dh_sparse_qsa_check
shared=$build/r3dh_sparse_qsa.so
image=freetoken:rocm10.1.0-transformers518-final-gh-proxy-20261026
base=(docker run --rm --user 1001:1001 -v "$workspace:$workspace" -w "$repo")
compiler=("${base[@]}" --entrypoint /opt/rocm/llvm/bin/clang++ "$image")
runner=("${base[@]}" --entrypoint "$binary" "$image")
flags=(-std=c++17 -O2 -fPIC -ffp-contract=off -fno-fast-math -Iabi -Ilibr3d -Ilibref)
objects=("$build/r3dh_sparse_qsa_host.o" "$build/r3dh_sparse_qsa_index.o" "$build/r3dh_sparse_qsa_device.o")
if [[ "$mode" == build || "$mode" == all ]]; then
    "${compiler[@]}" --version > "$logs/r3dh_sparse_qsa-build.log" 2>&1
    printf 'FLAGS %s --offload-arch=gfx1100\n' "${flags[*]}" >> "$logs/r3dh_sparse_qsa-build.log"
    for source in ref_attn ref_qsa; do
        "${compiler[@]}" "${flags[@]}" -c "libref/$source.cpp" -o "$build/r3dh_sparse_qsa_$source.o" >> "$logs/r3dh_sparse_qsa-build.log" 2>&1
    done
    "${compiler[@]}" "${flags[@]}" -c libr3d/r3dh_sparse_qsa_host.cpp -o "${objects[0]}" >> "$logs/r3dh_sparse_qsa-build.log" 2>&1
    for source in index device; do
        "${compiler[@]}" "${flags[@]}" --offload-arch=gfx1100 -c "libr3d/r3dh_sparse_qsa_$source.hip" -o "$build/r3dh_sparse_qsa_$source.o" >> "$logs/r3dh_sparse_qsa-build.log" 2>&1
    done
    "${compiler[@]}" "${flags[@]}" --offload-arch=gfx1100 --hip-link -shared "${objects[@]}" -lamdhip64 -o "$shared" >> "$logs/r3dh_sparse_qsa-build.log" 2>&1
    "${compiler[@]}" "${flags[@]}" --offload-arch=gfx1100 -c libr3d/r3dh_attn_qsa_state.hip -o "$build/r3dh_sparse_qsa_test_state.o" >> "$logs/r3dh_sparse_qsa-build.log" 2>&1
    "${compiler[@]}" "${flags[@]}" --offload-arch=gfx1100 libr3d/r3dh_sparse_qsa_check.hip "${objects[@]}" \
        "$build/r3dh_sparse_qsa_test_state.o" "$build/r3dh_sparse_qsa_ref_attn.o" "$build/r3dh_sparse_qsa_ref_qsa.o" \
        -o "$binary" >> "$logs/r3dh_sparse_qsa-build.log" 2>&1
    for policy in fast-math finite-math-only; do
        if "${compiler[@]}" -std=c++17 "-f$policy" -fsyntax-only -x c++ -Iabi -Ilibr3d libr3d/r3dh_sparse_qsa.h > "$logs/r3dh_sparse_qsa-$policy-reject.log" 2>&1; then
            printf 'FAIL accepted %s\n' "$policy" >&2; exit 1
        fi
        grep -q 'Strict floating point required' "$logs/r3dh_sparse_qsa-$policy-reject.log"
    done
    if nm -D "$shared" | grep -E 'ref_attn|ref_qsa|r3dh_attn_qsa|cpu_fallback'; then printf 'FAIL production CPU or test dependency\n' >&2; exit 1; fi
    nm -D --defined-only "$shared" | grep ' T r3dh_sparse_qsa_' > "$logs/r3dh_sparse_qsa-exports.log"
    [[ $(wc -l < "$logs/r3dh_sparse_qsa-exports.log") == 11 ]]
    sha256sum "$repo"/libr3d/r3dh_sparse_qsa* "$shared" "$binary" > "$logs/r3dh_sparse_qsa-hashes.log"
    printf 'SPARSE_QSA_BUILD_PASS production_exports=11 cpu_oracle_linked=0\n'
fi
if [[ "$mode" == host || "$mode" == all ]]; then
    "${runner[@]}" --host > "$logs/r3dh_sparse_qsa-host.log" 2>&1
    tail -n 2 "$logs/r3dh_sparse_qsa-host.log"
fi
if [[ "$mode" == gpu || "$mode" == all ]]; then
    physical=${R3DH_SPARSE_QSA_GPU:-1}
    case "$physical" in
        1) render=129; bdf=0000:23:00.0 ;;
        4) render=132; bdf=0000:83:00.0 ;;
        *) printf 'Only physical GPU1 or GPU4 is allowed\n' >&2; exit 2 ;;
    esac
    lock=/home/zihaomu/bigssd/oj_system/var/locks/gpu-renderD$render.lock
    container=r3dh-sparse-qsa-gpu-$$
    exec 8<"$lock"
    trap 'timeout --foreground --kill-after=2s 5s docker rm -f "$container" >/dev/null 2>&1 || true' EXIT
    if {
        flock -n -E 75 8 &&
        [[ $(basename "$(readlink -f "/sys/class/drm/renderD$render/device")") == "$bdf" ]] &&
        if fuser "/dev/dri/renderD$render" > /dev/null 2>&1; then printf 'GPU%s has another render-node user; refusing\n' "$physical" >&2; exit 75; fi &&
        printf 'LOCK physical=%s renderD%s bdf=%s\n' "$physical" "$render" "$bdf" &&
        timeout --foreground --kill-after=2s 50s "${base[@]}" --name "$container" --device /dev/kfd --device /dev/dri --group-add 993 --group-add 44 \
            --security-opt seccomp=unconfined -e "ROCR_VISIBLE_DEVICES=$physical" -e HIP_VISIBLE_DEVICES=0 \
            -e "R3DH_SPARSE_QSA_EXPECT_BDF=$bdf" \
            --entrypoint "$binary" "$image" --gpu
    } > "$logs/r3dh_sparse_qsa-gpu.log" 2>&1; then
        tail -n 2 "$logs/r3dh_sparse_qsa-gpu.log"
    else
        status=$?; printf 'SPARSE_QSA_GPU_EXIT=%d (75 means selected GPU busy)\n' "$status" >&2; exit "$status"
    fi
fi
printf 'SPARSE_QSA_TEST_PASS mode=%s logs=%s\n' "$mode" "$logs"
