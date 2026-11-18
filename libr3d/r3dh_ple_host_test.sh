#!/usr/bin/env bash
set -euo pipefail
source_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
workspace=$(cd -- "$source_dir/.." && pwd)
compiler=${CXX:-/opt/rocm/llvm/bin/clang++}
provider=${1:-$workspace/build-r3d-gfx1100/radiance_home/kernels/libavx.so}
model=${2:-$workspace/models/qwen3.8-next-flash-iqr-v4.rad}
build=${R3DH_PLE_HOST_BUILD:-$workspace/build-r3d-bootstrap/r3dh_ple_host}
if ! command -v "$compiler" >/dev/null 2>&1; then
    if [[ ${R3DH_PLE_HOST_IN_SDK:-0} == 1 ]]; then
        printf 'PLE_HOST_SUITE_FAIL missing SDK compiler: %s\n' "$compiler" >&2
        exit 127
    fi
    sdk=${R3DH_PLE_HOST_SDK_IMAGE:-freetoken:rocm10.1.0-transformers518-final-gh-proxy-20261026}
    printf 'PLE_HOST_SDK image=%s mount=%s GPU_devices=0\n' "$sdk" "$workspace"
    exec docker run --rm --user "$(id -u):$(id -g)" --network none \
        --security-opt seccomp=unconfined \
        -e R3DH_PLE_HOST_IN_SDK=1 -e CXX="$compiler" \
        -e R3DH_PLE_HOST_BUILD="$build" \
        -e R3DH_PLE_HOST_CORE_BUILD="${R3DH_PLE_HOST_CORE_BUILD:-$workspace/build-r3d-gfx1100}" \
        -v "$workspace:$workspace" -w "$source_dir" \
        --entrypoint /bin/bash "$sdk" \
        "$source_dir/libr3d/r3dh_ple_host_test.sh" "$provider" "$model"
fi
logs=$workspace/results/r3d-iqr/implementation/logs
mkdir -p -- "$build" "$logs"
provider=$(realpath -- "$provider")
home=$(dirname -- "$(dirname -- "$provider")")
log=$logs/r3dh_ple_host-$(date -u +%Y%m%dT%H%M%SZ)-$$.log
exec > >(tee "$log") 2>&1
trap 'printf "PLE_HOST_SUITE_FAIL exit=%s line=%s\n" "$?" "$LINENO"' ERR
printf "PLE_HOST_SUITE_BEGIN compiler=%s provider=%s model=%s GPU_used=0\n" "$compiler" "$provider" "$model"
"$compiler" --version
cd -- "$source_dir"
flags=(-std=c++20 -O2 -Wall -Wextra -Werror -Iabi -Ilibr3d)
host=(libr3d/r3dh_ple_host.cpp libr3d/r3dh_ple_host_registry.cpp libr3d/r3dh_ple_host_hint.cpp)
"$compiler" "${flags[@]}" -fPIC -shared "${host[@]}" libr3d/r3dh_ple_host_plugin.cpp -Wl,-z,defs -Wl,-Bsymbolic -ldl -pthread -o "$build/r3dh_ple_host.so"
ln -sfn -- "$provider" "$build/libavx.so"
for fixture in 1 2 3 4 5 6 7 8; do
    "$compiler" "${flags[@]}" -fPIC -shared -DR3DH_PLE_HOST_FIXTURE=$fixture libr3d/r3dh_ple_host_provider_fixture.cpp -Wl,-z,defs -o "$build/r3dh_ple_host_refusal$fixture.so"
done
"$compiler" "${flags[@]}" libr3d/r3dh_ple_host_loader_check.cpp -ldl -pthread -o "$build/r3dh_ple_host_loader_check"
core_build=${R3DH_PLE_HOST_CORE_BUILD:-$workspace/build-r3d-gfx1100}
"$compiler" "${flags[@]}" -Icore libr3d/r3dh_ple_host_registry_check.cpp \
    -Wl,--start-group "$core_build/core/librad_core.a" "$core_build/core/librad_json.a" \
    "$core_build/core/librad_simd.a" "$core_build"/vendor/*.a -Wl,--end-group \
    -L/opt/rocm/lib -Wl,-rpath,/opt/rocm/lib -lamdhip64 -lhsa-runtime64 -ldl -pthread \
    -o "$build/r3dh_ple_host_registry_check"
for checker in check cpu_check file_check sparse_check model_check; do
    "$compiler" "${flags[@]}" -DR3DH_PLE_HOST_CPU_ONLY "libr3d/r3dh_ple_host_$checker.cpp" -L"$build" -l:r3dh_ple_host.so -Wl,-rpath,"$build" -ldl -pthread -o "$build/r3dh_ple_host_$checker"
done
for library in "$provider" "$build/r3dh_ple_host.so"; do
    linkage=$(ldd -r "$library" 2>&1)
    printf "%s\n" "$linkage"
    if [[ "$linkage" == *"not found"* || "$linkage" == *"undefined symbol"* ]]; then exit 1; fi
done
"$build/r3dh_ple_host_loader_check" "$build/r3dh_ple_host.so" "$provider" "$home" "$build"
"$build/r3dh_ple_host_registry_check" "$build/r3dh_ple_host.so" "$provider"
"$build/r3dh_ple_host_check" "$provider"
"$build/r3dh_ple_host_cpu_check" "$provider"
"$build/r3dh_ple_host_file_check" "$provider" "$build"
"$build/r3dh_ple_host_sparse_check" "$provider" "$build"
"$build/r3dh_ple_host_model_check" "$provider" "$model"
sha256sum "$provider" "$build/r3dh_ple_host.so" libr3d/r3dh_ple_host*
printf "PLE_HOST_SUITE_PASS build=%s log=%s public_HOST=1 internal_rows=3 DEVICE=0 GPU_used=0 full_model=not_tested\n" "$build" "$log"
