/* aql.h -- declarations shared by the AQL backend's translation units.
 *
 * aql_code.cpp reads code objects (offload bundle, ELF note, msgpack metadata), aql.cpp owns the
 * queues and all submission, and aql_blit.hip holds the backend's own copy and fill kernels.
 * aql_blit.hip is compiled as HIP, so this header must not pull in HSA.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace rad {
namespace aql {

/* ------------------------------------------------------------------ code objects */

/* One kernel argument as laid out in the code object's metadata. `hidden` names the implicit
 * arguments the compiler appends; explicit parameters map in order onto hipLaunchKernel's
 * `kernelParams`. */
enum class Hidden : uint8_t {
    None,           /* an explicit parameter */
    Pad,            /* hidden_none: bytes the compiler reserved and reads nothing from */
    BlockCountX, BlockCountY, BlockCountZ,
    GroupSizeX, GroupSizeY, GroupSizeZ,
    RemainderX, RemainderY, RemainderZ,
    GlobalOffsetX, GlobalOffsetY, GlobalOffsetZ,
    GridDims,
    DynamicLds,
    Unsupported,    /* anything else: a kernel that wants it is refused by name */
};

struct KernelArg {
    uint32_t offset = 0;
    uint32_t size   = 0;
    Hidden   hidden = Hidden::None;
    std::string kind;              /* the metadata's value_kind, for the refusal message */
};

struct KernelMeta {
    std::string symbol;            /* "<name>.kd" */
    uint32_t kernarg_size = 0;
    uint32_t group_size   = 0;     /* static LDS */
    uint32_t private_size = 0;     /* static scratch */
    bool     dynamic_stack = false;
    std::vector<KernelArg> args;
};

/* Selects the card's code object from a clang offload bundle, or fails with `*why` naming what
 * the bundle holds and what the card accepts. `isas` lists the card's accepted targets in the
 * runtime's order of preference ("amdgcn-amd-amdhsa--gfx1036", then the family's
 * "amdgcn-amd-amdhsa--gfx10-3-generic"), each with its modes ("gfx942:sramecc+:xnack-"). The
 * bundle has no total length, so it is walked entry by entry. Compressed bundles are refused;
 * nothing in this tree builds one. */
bool bundle_code_object(const void* bundle, const std::vector<std::string>& isas, const void** out,
                        size_t* out_size, std::string* why);

/* Every kernel described in an AMDGPU code object's metadata note. */
bool code_object_kernels(const void* elf, size_t size, std::vector<KernelMeta>* out,
                         std::string* why);

/* ------------------------------------------------------------------ fence policy
 *
 * Extra fence scope for the next dispatch on this thread, beyond the agent scope every kernel
 * gets. Set by the backend around its own blits; cleared by the dispatch that consumes it. */
enum : uint8_t { kFenceAcquireSystem = 1, kFenceReleaseSystem = 2 };
void set_next_fence(uint8_t f);

/* ------------------------------------------------------------------ test hooks */

/* Whether the signal behind a rad_ride token could be handed out again now (tests/device_test.cpp). */
bool ride_reusable(void* token);

/* Enqueue the same wait as rad_ride_wait; ownership stays with the caller, including on error. */
int ride_wait_keep(void* stream, void* token);

#ifdef RAD_AQL_RETIRE_DIAGNOSTICS
struct RetireCounts {
    uint64_t gets = 0, creates = 0, retirements = 0, value_reads = 0;
    uint64_t fresh_reads = 0, fresh_nonzero = 0, queue_reads = 0, fresh_queue_reads = 0;
    uint64_t waits = 0, retiring_peak = 0;
};
void retire_count_begin();
RetireCounts retire_count_end();
#endif

/* `pairs` launches of two chained kernels onto `stream`, one behind the other: the first stores
 * i + 1 over the 64 words at `x`; the second loads x[0] the way a kernel loads a value all its
 * lanes share (a scalar load) and each of its `blocks` blocks stores what it saw, at
 * seen[i * blocks + block]. Every reader must have seen the store in front of it
 * (tests/device_test.cpp). RAD_E_UNSUPPORTED in a build without the backend. */
int chain_probe(void* stream, unsigned* x, unsigned* seen, unsigned pairs, unsigned blocks);

/* `n` chained launches onto `stream`, launch i storing i + 1 at seen[i]: each takes where to
 * store from its own arguments, so a launch that read another's leaves a word unwritten
 * (tests/device_test.cpp). RAD_E_UNSUPPORTED in a build without the backend. */
int chain_marks(void* stream, unsigned* seen, unsigned n);

/* `n` chained launches onto `stream`, each a card-side copy of `bytes` (a multiple of 16) from
 * `src` to `dst` that also stores at seen[i] whether it found a chain record in its argument
 * slot: i + 1 if every block did, that with the top bit set if one did not. A block that found
 * none still counts, so the launches behind it are not left waiting (tests/device_test.cpp).
 * RAD_E_UNSUPPORTED in a build without the backend. */
int chain_asks(void* stream, const void* src, void* dst, uint64_t bytes, unsigned* seen, unsigned n);

}  // namespace aql
}  // namespace rad

struct RadCopyRange;

/* Blit kernel launchers, defined in aql_blit.hip. Each issues one kernel onto `stream` through
 * hipLaunchKernel, which the backend intercepts. The caller guarantees alignment: the 16-byte
 * forms need both addresses and the length 16-byte aligned. */
extern "C" {
/* `link`: either side is pinned host memory, so use one narrow workgroup instead of a full grid
 * (see aql_blit.hip). */
int rad_aql_blit_copy(void* dst, const void* src, uint64_t bytes, void* stream, bool link = false);
int rad_aql_blit_copy2d(void* dst, uint64_t dpitch, const void* src, uint64_t spitch,
                        uint64_t width, uint64_t height, void* stream);
int rad_aql_blit_fill(void* dst, int value, uint64_t bytes, void* stream);
/* rad_memcpy_ranges_async's kernel: `prefix` a multiple of 16, both bases 16-byte aligned. */
int rad_aql_blit_ranges(void* dst, const void* src, uint64_t prefix, const RadCopyRange* ranges,
                        int n, void* stream);
/* aql::chain_probe's two kernels, aql::chain_marks's one and aql::chain_asks's one. */
int rad_aql_probe_store(unsigned* x, unsigned v, void* stream);
int rad_aql_probe_load(const unsigned* x, unsigned* seen, unsigned blocks, void* stream);
int rad_aql_probe_mark(unsigned* seen, unsigned v, void* stream);
int rad_aql_probe_ask(const void* src, void* dst, uint64_t bytes, unsigned* seen, unsigned v,
                      unsigned long long* ctr, void* stream);
}
