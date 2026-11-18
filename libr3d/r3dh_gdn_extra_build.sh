#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
workspace=$(dirname "$root")
image=${R3DH_GDN_EXTRA_SDK:-freetoken:rocm10.1.0-transformers518-final-gh-proxy-20261026}
if [[ ${1:-} != --inside ]]; then
    exec docker run --rm --user 1001:1001 -v "$workspace:$workspace" -w "$root" \
        --entrypoint /bin/bash "$image" "$root/libr3d/r3dh_gdn_extra_build.sh" --inside "${@}"
fi
shift
cd "$root"
build="$workspace/build-r3d-bootstrap"
mkdir -p "$build"
cxx=/opt/rocm/llvm/bin/clang++
flags=(-std=c++20 -O2 -fPIC -fno-fast-math -ffp-contract=off -ffunction-sections -fdata-sections -Iabi -Ilibr3d)
for unit in strict chunk_strict; do
    "$cxx" "${flags[@]}" --offload-arch=gfx1100 -c "libr3d/r3dh_gdn_extra_${unit}.hip" -o "$build/r3dh_gdn_extra_${unit}.o"
done
"$cxx" --hip-link -shared -Wl,--gc-sections "$build/r3dh_gdn_extra_strict.o" \
    "$build/r3dh_gdn_extra_chunk_strict.o" -lamdhip64 -o "$build/libr3dh_gdn_extra_strict.so"
for unit in gdn quant; do
    "$cxx" "${flags[@]}" -Ilibref -c "libref/ref_${unit}.cpp" -o "$build/r3dh_gdn_extra_ref_${unit}.o"
done
"$cxx" "${flags[@]}" --offload-arch=gfx1100 -Ilibref -c libr3d/r3dh_gdn_extra_check.hip -o "$build/r3dh_gdn_extra_check.o"
"$cxx" --hip-link -Wl,--gc-sections "$build/r3dh_gdn_extra_check.o" \
    "$build/r3dh_gdn_extra_strict.o" "$build/r3dh_gdn_extra_chunk_strict.o" \
    "$build/r3dh_gdn_extra_ref_gdn.o" "$build/r3dh_gdn_extra_ref_quant.o" -lamdhip64 -o "$build/r3dh_gdn_extra_check"
if [[ ${1:-} == --fast-compile ]]; then
    for source in libr3d/r3dh_gdn_extra_gdn_*.hip; do
        "$cxx" "${flags[@]}" --offload-arch=gfx1100 -c "$source" -o "$build/$(basename "${source%.hip}").o"
    done
fi
printf 'PASS build native=gfx1100 contract=off strict_library_and_independent_libref_checker\n'
