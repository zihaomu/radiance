#pragma once
#include "rad_plugin.h"

extern "C" {
int r3dh_gdn_extra_conv_prep_strict(const RadArgs*, RadStream);
int r3dh_gdn_extra_conv_update_strict(const RadArgs*, RadStream);
int r3dh_gdn_extra_recurrent_strict(const RadArgs*, RadStream);
int r3dh_gdn_extra_conv_recurrent_update_had_strict(const RadArgs*, RadStream);
int r3dh_gdn_extra_had_quant_strict(const RadArgs*, RadStream);
int r3dh_gdn_extra_validate(const RadArgs*, int);
int64_t r3dh_gdn_extra_scratch(const RadArgs*);
int r3dh_gdn_extra_kkt_strict(const RadArgs*, RadStream);
int r3dh_gdn_extra_chunk_strict(const RadArgs*, RadStream);
}
