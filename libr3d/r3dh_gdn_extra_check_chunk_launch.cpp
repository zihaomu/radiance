#include "r3d_model_extra_registry.cpp"
extern "C" int r3dh_gdn_extra_check_kkt_launch(const RadArgs* args, RadStream stream) {
    return chunk_launch<true>(args, stream);
}
extern "C" int r3dh_gdn_extra_check_scan_launch(const RadArgs* args, RadStream stream) {
    return chunk_launch<false>(args, stream);
}
