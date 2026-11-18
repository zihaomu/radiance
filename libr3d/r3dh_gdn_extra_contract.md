# GDN Extra Local Contract And Qualification

Verified 2026-11-17 on physical GPU2, gfx1100, PCI 0000:43:00.0.
Only `r3dh_gdn_extra*` source files belong to this task. No CMake, rows,
core, registry, or other agent files were edited. This is a standalone native
GPU implementation and checker, not full-model integration or a benchmark.

## Completed Files

- `r3dh_gdn_extra_check.hip`: corrected the initial fused anchor acceptance
  count, explicit anchor contract regressions, strict chunk rejection tests,
  and absent-initial-state coverage.
- `r3dh_gdn_extra_fp8_frag.h`: missing private fragment permutation dependency;
  4096 positions across four tiles checked for range and bijection.
- `r3dh_gdn_extra_test.sh`: standalone build plus two fresh GPU2 checker
  processes, production `flock -n`, read-open lock, exit 75 when busy.
- `r3dh_gdn_extra_contract.md`: this handoff.
- Existing `r3dh_gdn_extra.h`, `r3dh_gdn_extra_strict.hip`,
  `r3dh_gdn_extra_chunk_strict.hip`, and `r3dh_gdn_extra_build.sh` were built
  and validated without modification by this continuation.

## Anchor Failure Resolved

The previous status -12 is `RAD_E_INVAL`, not an ABI-version failure.
`ref_gdn_conv_recurrent_update` first calls convolution update, which reads
history at `num_accepted - 1`; zero is invalid for that fused operation.
The previous anchor fixture passed zero. It now starts at one, with zeroed
replay-factor blocks. The standalone recurrent anchor form can accept zero;
that does not make zero valid for the fused convolution/recurrent schema.

Four fixtures explicitly confirm libref's zero-count rejection, a local
five-column anchor-index rejection, and a non-F32 recurrent-state rejection.
Anchor columns 0..2 are committed-state pages; columns 3..5 are replay-factor
pages. Tests use distinct pages, six columns, and the dense 128x128 F32
per-head factor layout. Each head holds at most 16 replay factors with
stride 648 floats and stamp offset 10368. Moving page windows are not tested.

## Symbols And Operand Order

The following are C exports in `build-r3d-bootstrap/libr3dh_gdn_extra_strict.so`.
Each listed suffix has prefix `r3dh_gdn_extra_`. Lists are zero-based operand
order, not additional schema definitions. No canonical schemas were changed.

| Suffix | Schema / purpose | Ordered operands |
| --- | --- | --- |
| `conv_update_strict` | `gdn_conv_update` | x, conv_w, conv_b, conv_state, conv_idx, num_accepted, cu_seqlens, q, k, v |
| `conv_prep_strict` | `gdn_conv_prep` | x, conv_w, conv_b, A_log, dt_bias, conv_state, cu_seqlens, a, b, conv_idx, initialized, q, k, v, g, beta |
| `recurrent_strict` | `gdn_recurrent_update` | q, k, v, a, b, A_log, dt_bias, state, state_idx, num_accepted, cu_seqlens, z, norm_w, out |
| `conv_recurrent_update_had_strict` | `gdn_conv_recurrent_update_had` | x, conv_w, conv_b, conv_state, conv_idx, a, b, A_log, dt_bias, state, state_idx, num_accepted, cu_seqlens, z, norm_w, out, out_q, out_scale |
| `kkt_strict` | `gdn_kkt_solve` | k, beta, g, cu_seqlens, inverse |
| `chunk_strict` | `gdn_chunk_scan` | q, k, v, inverse, g, beta, initial_state, cu_seqlens, out, final_state, state_idx |
| `had_quant_strict` | private quantization helper | out, out_q, out_scale |
| `validate` | descriptor validation | RadArgs plus operation 0=conv_update, 1=conv_prep, 2=recurrent, 3=fused, 4=quant |
| `scratch` | fused temporary bytes | RadArgs; negative status on invalid input |
| `gdn_dims` | dimension query | int pointers: key=128, value=128, chunk=64 |

The quantization helper consumes a rank-three `[M,Hv,128]` BF16 view, while
canonical `had_quant_act_i8g` takes a rank-two activation view. Do not register
the helper directly as that canonical op without a view adapter. The checker
uses the canonical 2D libref view and compares the same physical output.

## Constraints

- Current model: Hk=16, Hv=48; TP2 rank: Hk=8, Hv=24. Only these two pairs
  and Hv/Hk=3 are accepted. Key/value dimensions 128; convolution width 4;
  chunk 64; M=1..2048; 1..64 sequences.
- x, conv_w, optional conv_b, q/k/v, a/b, z and output are BF16. A_log,
  dt_bias, norm_w, decay g, beta, recurrent state, and chunk state are F32.
  Indices, acceptance counts, initialized flags and cu_seqlens are I32.
- x is `[M,at_least_C]`, C=(2*Hk+Hv)*128: 5120 per TP2 rank or 10240 full.
  q/k are `[M,Hk,128]`, v/out `[M,Hv,128]`; scalar gates `[M,Hv]`;
  conv weights `[C,4]`; norm weights `[128]`; inverse `[M,Hv,64]` BF16.
- Recurrent/chunk state is `[slots,Hv,128,128]` F32, last strides 128,1.
  Outer row/head/slot padding is supported and tested. Output integer codes
  are row-major I8 `[M,Hv*128]`, scales F32 `[M,Hv*128/group]`, group 128/256.
- Fused decode/conv-update q_len=1..8; standalone recurrent q_len=1..16.
  Decode convolution depth is at least q_len+2; prefill depth at least 3.
  Per-candidate state indices have q_len columns; anchor indices at least six.
- Fused convolution acceptance must be at least one with a valid history
  window. Device index values and monotone cumulative lengths remain caller
  preconditions; descriptor validation does not copy device metadata to CPU.
  All referenced pages must exist and state/factor page aliases must obey the
  upstream anchor contract. Unsupported device metadata is not qualified.
- Chunk initial state may be absent, meaning zero. Optional conv bias and
  initialized flags follow the local validator. Finite scale, positive eps
  and l2_eps, activation silu/sigmoid only; no dtype or geometry fallback.
- Fused scratch = round_up(M*(2*Hk+Hv)*256,256) bytes; pointer must be
  16-byte aligned. Scratch and buffers remain live until the stream completes.
- Strict floating-point ordering is intentional, including BF16 RNE and the
  integer FWHT order. No tolerance relaxation and no CPU execution fallback.

## Build And Replay

From the worktree on the host:

```bash
bash libr3d/r3dh_gdn_extra_build.sh
bash libr3d/r3dh_gdn_extra_test.sh
bash libr3d/r3dh_gdn_extra_build.sh --fast-compile
```

Existing SDK image:
`freetoken:rocm10.1.0-transformers518-final-gh-proxy-20261026`.
Compiler `/opt/rocm/llvm/bin/clang++`, AMD clang 24.0.0git, revision
`d6f6cb691863e9258034ab48c1dfce21689d5606`.

Flags: `-std=c++20 -O2 -fPIC -fno-fast-math -ffp-contract=off
-ffunction-sections -fdata-sections -Iabi -Ilibr3d`; HIP units additionally
`--offload-arch=gfx1100`; libref/checker also `-Ilibref`. Link
`--hip-link -Wl,--gc-sections -lamdhip64`, plus `-shared` for the library.
No package installation or global environment changes. No fast-math.

Strict production linkage contains only strict/chunk-strict objects; libref
`ref_gdn.cpp` and `ref_quant.cpp` are linked into the independent checker only.
Dynamic-symbol audit found no libref dependency in the strict shared library.
Offload-bundle audit found gfx1100 for both HIP translation units.

GPU runs read-open and nonblocking-lock
`/home/zihaomu/bigssd/oj_system/var/locks/gpu-renderD130.lock`, expose
`/dev/kfd` and `/dev/dri`, set ROCR_VISIBLE_DEVICES=2 and
HIP_VISIBLE_DEVICES=0, assert one visible gfx1100 device, and use non-default
streams. Exposing only renderD130 hides GPU2 under this ROCR enumeration and
is not the verified container recipe. Locks release when the runner exits.

## Actual Results

Durable logs in `results/r3d-iqr/implementation/logs/`, relative to workspace:

- `r3dh-gdn-extra-20261117T044924Z-build.log`: strict build PASS.
- `r3dh-gdn-extra-20261117T044924Z-gpu.log`: two fresh processes, both PASS;
  54 decode steps, including 24 anchor steps; 24 anchor contract assertions;
  10 prefill/KKT/scan pipelines; 30 chunk contract assertions. Another 36
  existing shallow-conv-state/short-scratch rejection assertions also pass.
- Per process: nine decode fixtures, three changed-input stateful steps each;
  all nine capture/instantiate/execute a graph on step one. TP2 tests M=1,
  ragged two-sequence M=4/q_len=3, and anchor M=8/q_len=8. Both 128/256 groups,
  silu/sigmoid. Full geometry also tested at M=1/group256/per_candidate.
- Per process: TP2 prefill M=1/63/64/65 and full geometry M=1; three warm and
  two absent-initial-state chunk cases. BF16 inverse/output and entire padded
  F32 final-state buffers compare bit-for-bit to libref. All guards pass.
- `r3dh-gdn-extra-contract-gpu.log`: earlier single-process strengthened
  suite PASS; 27 decode steps, 12 anchor assertions, five prefill pipelines,
  15 chunk assertions. Do not add reruns together as unique test coverage.
- `r3dh-gdn-extra-native-carried-build.log`: four carried optimized HIP
  translation units compile for gfx1100 after the missing private header fix.
- `r3dh-gdn-extra-fragment-mapping.log`: 4096 mapped byte positions PASS.

## Pending / Registration Boundary

- No requested local anchor or strict chunk test remains blocked.
- Main CMake/rows/registry integration is intentionally left to its owner.
  This library has C entry points, not a standalone RadKernelLib registration.
- Carried `r3dh_gdn_extra_gdn_*.hip` optimized variants are compile-only,
  not numerically qualified or registered. Keep them out of strict linkage:
  their chunk object also defines `r3dh_gdn_extra_gdn_dims`.
- No real model weights or TP2 end-to-end execution, performance claim,
  moving anchor page windows, empty sequences, multi-sequence chunk scan,
  chunk graph replay, state aliasing, or full M<=2048 sweep was tested.
  Standalone recurrent-only zero acceptance also remains unqualified.
