#include "r3dh_moe_mixed_model.h"
int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::runtime_error("usage: mixed_model_check MODEL");
        Model model(argv[1]);
        for (int layer : {34, 44, 47}) {
            int regular = 0, protected_count = 0;
            for (const auto& entry : model.entries) {
                const std::string name = model.name(entry);
                int named_layer = -1, expert = -1, consumed = 0;
                if (std::sscanf(name.c_str(), "blk.%d.ffn_gate_up_exps.%d.weight%n",
                    &named_layer, &expert, &consumed) != 2 || !consumed
                    || size_t(consumed) != name.size() || named_layer != layer) continue;
                if (entry.enc >= model.encodings.size()) throw std::runtime_error("encoding bounds");
                const auto& encoding = model.encodings[entry.enc];
                const auto* codes = rad_enc_plane(&encoding, "codes");
                if (!codes) throw std::runtime_error("missing codes");
                if (codes->dtype == RAD_BF16) {
                    ++protected_count;
                    std::printf("PROTECTED layer=%d expert=%d shape=%lld,%lld planes=%u offset=%llu bytes=%llu\n",
                        layer, expert, (long long)entry.shape[0], (long long)entry.shape[1],
                        entry.n_planes, (unsigned long long)entry.offset, (unsigned long long)entry.bytes);
                } else {
                    if (!regular) std::printf("REGULAR layer=%d dtype=%u scheme=%s transform=%s\n",
                        layer, codes->dtype, encoding.scheme, encoding.transform);
                    ++regular;
                }
            }
            std::printf("COUNTS layer=%d regular=%d protected=%d\n", layer, regular, protected_count);
            if (protected_count < 1 || protected_count > 10 || regular + protected_count != 512)
                throw std::runtime_error("mixed expert counts");
            std::printf("MIXED_MODEL_PASS layer=%d regular=%d protected=%d\n", layer, regular, protected_count);
        }
    } catch (const std::exception& error) { std::fprintf(stderr, "%s\n", error.what()); return 1; }
}
