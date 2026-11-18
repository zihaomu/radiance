#pragma once

#include "rad_abi.h"
#include "rad_plugin.h"

struct R3dhPleHost {
    RadKernelInfo rows[2];
};

extern "C" {
int r3dh_ple_host_bind(R3dhPleHost*, int, const RadKernelInfo*, const RadKernelInfo*);
int r3dh_ple_host_schema_count();
const RadOpSchema* r3dh_ple_host_schema_at(int);
const RadKernelInfo* r3dh_ple_host_kernel_at(const R3dhPleHost*, int);
int r3dh_ple_host_open(const char*);
int r3dh_ple_host_close();
int r3dh_ple_host_registry_count();
const RadKernelInfo* r3dh_ple_host_registry_at(int);
const RadKernelInfo* r3dh_ple_host_hint_kernel();
int r3dh_ple_host_ngram_hint(const RadArgs*, RadStream);
int r3dh_ple_host_device_kernel_count();
const RadKernelInfo* r3dh_ple_host_device_kernel_at(int);
int r3dh_ple_host_gain_layout(const RadParam*, int, int, const RadEncoding*,
                             const int*, const RadTensor*, int, RadLayout*);
int r3dh_ple_host_gain_relayout(const RadParam*, int, int, const RadEncoding*,
                               const int*, const RadTensor*, int, void*, int64_t);
int r3dh_ple_host_gain_unrelayout(const RadParam*, int, int, const RadEncoding*,
                                 const int*, const void*, int64_t, const RadTensor*, int);
}
