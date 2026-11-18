#!/usr/bin/env bash
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
workspace=$(dirname "$repo")
mode=${1:---all}
case "$mode" in --build|--host|--gpu|--all|--prepare-host|--prepare-gpu|--regression-gpu) ;; *) exit 2 ;; esac
build=$workspace/build-r3d-bootstrap/r3dh_attn_fused-build
logs=$workspace/results/r3d-iqr/implementation/logs
mkdir -p "$build" "$logs"
binary=$build/r3dh_attn_fused_check
shared=$build/r3dh_attn_fused.so
image=radiance-r3d-dev:rocm10.1-ffmpeg
base=(docker run --rm --user 1001:1001 -v "$workspace:$workspace" -w "$repo")
compiler=("${base[@]}" --entrypoint /opt/rocm/llvm/bin/clang++ "$image")
flags=(-std=c++17 -O2 -fPIC -fno-fast-math -ffp-contract=off -fdenormal-fp-math=ieee
    -ffunction-sections -fdata-sections -Iabi -Ilibr3d -Ilibref)
if [[ "$mode" == --build || "$mode" == --all ]]; then
    exec 7>"$build/build.lock"
    flock -n -E 75 7
    buildlog=$logs/r3dh_attn_fused-build.log
    "${compiler[@]}" --version > "$buildlog" 2>&1
    docker image inspect "$image" --format '{{.Id}}' >> "$buildlog"
    printf 'FLAGS %s --offload-arch=gfx1100\n' "${flags[*]}" >> "$buildlog"
    sha256sum "$repo"/libr3d/r3dh_attn_fused* "$repo"/abi/*.h "$repo"/libref/*.h \
        "$repo"/libref/ref_{attn,qsa,quant}.cpp \
        "$repo"/libr3d/r3dh_attn_qsa{.h,_device.hip,_rope.hip,_state.hip,_check.hip} \
        "$repo"/libr3d/r3dh_sparse_qsa{.h,_internal.h,_host.cpp,_index.hip,_device.hip} \
        "$repo"/libr3d/r3dh_qwen_ops{.h,_quant.hip} > "$logs/r3dh_attn_fused-inputs.sha256"
    objects=()
    for source in r3dh_attn_fused.cpp r3dh_attn_fused_registry.cpp r3dh_attn_fused_native.hip \
        r3dh_attn_qsa_device.hip r3dh_attn_qsa_rope.hip r3dh_attn_qsa_state.hip \
        r3dh_sparse_qsa_host.cpp r3dh_sparse_qsa_index.hip r3dh_sparse_qsa_device.hip r3dh_qwen_ops_quant.hip; do
        object=$build/${source%.*}.o
        "${compiler[@]}" "${flags[@]}" --offload-arch=gfx1100 -c "libr3d/$source" -o "$object" >> "$buildlog" 2>&1
        objects+=("$object")
    done
    "${compiler[@]}" --offload-arch=gfx1100 --hip-link -shared "${objects[@]}" -lamdhip64 \
        -Wl,--no-undefined -Wl,--gc-sections -o "$shared" >> "$buildlog" 2>&1
    references=()
    for source in ref_attn ref_qsa ref_quant; do
        object=$build/$source.o
        "${compiler[@]}" "${flags[@]}" -c "libref/$source.cpp" -o "$object" >> "$buildlog" 2>&1
        references+=("$object")
    done
    "${compiler[@]}" "${flags[@]}" --offload-arch=gfx1100 libr3d/r3dh_attn_fused_check.hip \
        "${objects[@]}" "${references[@]}" -Wl,--gc-sections -o "$binary" >> "$buildlog" 2>&1
    for policy in fast-math finite-math-only; do
        if "${compiler[@]}" -std=c++17 "-f$policy" -fsyntax-only -x c++ -Iabi -Ilibr3d \
            libr3d/r3dh_attn_fused.h > "$logs/r3dh_attn_fused-$policy.log" 2>&1; then exit 1; fi
        grep -q 'Strict floating point required' "$logs/r3dh_attn_fused-$policy.log"
    done
    if nm -D "$shared" | grep -E 'ref_attn|ref_qsa|ref_quant|cpu_prepare|hipDeviceSynchronize|hipStreamSynchronize|hipMemcpy$'; then exit 1; fi
    nm -D --defined-only "$shared" | grep ' T r3dh_attn_fused_' > "$logs/r3dh_attn_fused-exports.log"
    [[ $(wc -l < "$logs/r3dh_attn_fused-exports.log") == 7 ]]
    sha256sum -c "$logs/r3dh_attn_fused-inputs.sha256" > "$logs/r3dh_attn_fused-inputs-verify.log"
    sha256sum "$repo"/libr3d/r3dh_attn_fused* "$shared" "$binary" > "$logs/r3dh_attn_fused-hashes.log"
    printf 'ATTN_FUSED_BUILD_PASS fresh_objects=10 strict=1 cpu_fallback=0 host_sync=0\n' | tee -a "$buildlog"
fi
if [[ "$mode" == --host || "$mode" == --all || "$mode" == --prepare-host ]]; then
    sha256sum -c "$logs/r3dh_attn_fused-inputs.sha256" > "$logs/r3dh_attn_fused-inputs-verify.log"
    host_mode=--host; if [[ "$mode" == --prepare-host ]]; then host_mode=--prepare-host; fi
    hostlog=$logs/r3dh_attn_fused-host.log
    if [[ "$mode" == --prepare-host ]]; then hostlog=$logs/r3dh_attn_fused-prepare-host.log; fi
    "${base[@]}" --entrypoint "$binary" "$image" "$host_mode" > "$hostlog" 2>&1
    tail -n 1 "$hostlog"
fi
if [[ "$mode" == --gpu || "$mode" == --all || "$mode" == --prepare-gpu || "$mode" == --regression-gpu ]]; then
    sha256sum -c "$logs/r3dh_attn_fused-inputs.sha256" > "$logs/r3dh_attn_fused-inputs-verify.log"
    acquired=0
    for physical in 6; do
        render=134; bdf=0000:c3:00.0
        exec 8<"/home/zihaomu/bigssd/oj_system/var/locks/gpu-renderD$render.lock"
        if ! flock -n -E 75 8; then exec 8<&-; continue; fi
        if [[ $(basename "$(readlink -f "/sys/class/drm/renderD$render/device")") != "$bdf" ]]; then exit 1; fi
        read -r busy < "/sys/class/drm/renderD$render/device/gpu_busy_percent"
        read -r vram < "/sys/class/drm/renderD$render/device/mem_info_vram_used"
        if [[ "$busy" != 0 || "$vram" -gt 67108864 ]] || fuser "/dev/dri/renderD$render" >/dev/null 2>&1; then
            flock -u 8; exec 8<&-; continue
        fi
        acquired=1
        container=r3dh-attn-fused-gpu-$$
        trap 'docker rm -f "$container" >/dev/null 2>&1 || true' EXIT
        gpulog=$logs/r3dh_attn_fused-gpu.log
        if [[ "$mode" == --prepare-gpu ]]; then gpulog=$logs/r3dh_attn_fused-prepare-gpu.log; fi
        if [[ "$mode" == --regression-gpu ]]; then gpulog=$logs/r3dh_attn_fused-regression-gpu.log; fi
        printf 'LOCK physical=%s renderD%s bdf=%s busy=%s vram=%s\n' "$physical" "$render" "$bdf" "$busy" "$vram" > "$gpulog"
        gpu_mode=--gpu; if [[ "$mode" == --prepare-gpu ]]; then gpu_mode=--prepare-gpu; fi
        if [[ "$mode" == --regression-gpu ]]; then gpu_mode=--regression-gpu; fi
        if "${base[@]}" --name "$container" --device /dev/kfd --device /dev/dri --group-add 993 --group-add 44 \
            --security-opt seccomp=unconfined -e "ROCR_VISIBLE_DEVICES=$physical" -e HIP_VISIBLE_DEVICES=0 \
            -e "R3DH_ATTN_FUSED_EXPECT_BDF=$bdf" --entrypoint "$binary" "$image" "$gpu_mode" >> "$gpulog" 2>&1; then
            printf 'ATTN_FUSED_GPU_EXIT=0 physical=%s\n' "$physical" >> "$gpulog"
        else status=$?; printf 'ATTN_FUSED_GPU_EXIT=%s\n' "$status" >> "$gpulog"; tail -n 8 "$gpulog"; exit "$status"; fi
        tail -n 2 "$gpulog"
        flock -u 8
        flock -n -E 75 8
        printf 'ATTN_FUSED_LOCK_REACQUIRE_PASS physical=%s\n' "$physical" | tee -a "$gpulog"
        flock -u 8
        sha256sum -c "$logs/r3dh_attn_fused-inputs.sha256" > "$logs/r3dh_attn_fused-inputs-verify.log"
        exec 8<&-
        break
    done
    if [[ "$acquired" != 1 ]]; then
        printf 'ATTN_FUSED_GPU_BUSY GPU6_only_no_fallback long_cases_run=0 gpu_launches=0 qualification=NOT_YET\n' | tee "$logs/r3dh_attn_fused-long-blocked.log" >&2
        exit 75
    fi
fi
