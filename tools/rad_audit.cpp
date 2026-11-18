#include "build/startup.h"
#include "rad_device.h"

#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>

using namespace rad;

namespace {

std::string csv(const std::string& value) {
    std::string result = "\"";
    for (char character : value) {
        if (character == '"') result += '"';
        result += character;
    }
    return result + '"';
}

}

int main(int argc, char** argv) {
    if (argc != 5) {
        std::fprintf(stderr, "usage: rad-audit MODEL HOME MAX_TOK SPEC_DEPTH\n");
        return 2;
    }
    RadFile file;
    if (file.open(argv[1]) < 0) return 1;
    LoadedPlugins plugins;
    if (rad_tools_load_plugins(argv[2], {"libr3d", "libr4d", "libavx", "libref"},
        file.str(file.header().arch_id), file.str(file.header().quant), &plugins) < 0) return 1;
    RadModelMeta metadata{};
    metadata.arch_id = file.str(file.header().arch_id);
    metadata.name = file.str(file.header().model_name);
    metadata.quant = file.str(file.header().quant);
    auto integer = [&](const char* key, int64_t fallback) {
        int64_t value = fallback; file.get_i(key, &value); return value;
    };
    auto floating = [&](const char* key, double fallback) {
        double value = fallback; file.get_f(key, &value); return static_cast<float>(value);
    };
    metadata.n_layers = integer("n_layers", 0); metadata.n_embd = integer("n_embd", 0);
    metadata.n_head = integer("n_head", 0); metadata.n_head_kv = integer("n_head_kv", 0);
    metadata.head_dim = integer("head_dim", 0); metadata.n_ff = integer("n_ff", 0);
    metadata.n_vocab = integer("n_vocab", 0); metadata.n_ctx_train = integer("n_ctx_train", 0);
    metadata.n_expert = integer("n_expert", 0); metadata.n_expert_used = integer("n_expert_used", 0);
    metadata.n_expert_shared = integer("n_expert_shared", 0);
    metadata.rms_eps = floating("rms_eps", 1e-6); metadata.rope_theta = floating("rope_theta", 10000);
    metadata.rope_scale = 1;
    std::vector<std::string> keys, values;
    std::vector<const char*> key_pointers, value_pointers;
    for (int64_t index = 0; index < file.meta_count(); ++index) {
        const RadFileKV& item = file.meta_at(index);
        if (item.type != RAD_P_STR) continue;
        keys.push_back(file.str(item.key)); values.push_back(file.str(item.v.s));
    }
    for (auto& key : keys) key_pointers.push_back(key.c_str());
    for (auto& value : values) value_pointers.push_back(value.c_str());
    metadata.n_kv = keys.size(); metadata.kv_key = key_pointers.data(); metadata.kv_val = value_pointers.data();
    Registry& registry = rad_tools_registry();
    registry.set_device_fact(RAD_FACT_P2P, 1);
    const Plugin* architecture = registry.plugin(plugins.arch_plugin);
    if (!architecture || !architecture->arch_declare) return 1;
    std::puts("rank,declare_status,op_index,op,geometry,band,device_plugin,device_kernel,host_plugin,missing_device,weights,encodings");
    bool complete = true;
    for (int rank = 0; rank < 2; ++rank) {
        RadBuildCtx context{};
        context.rank = rank; context.world_size = 2; context.max_tok = std::atoi(argv[3]);
        context.max_seqs = 8; context.max_ctx = 200000; context.max_spec = std::atoi(argv[4]);
        context.max_enc_patches = 2048; context.kv_dtype = RAD_I8; context.scope = "";
        if (context.max_tok < 1 || context.max_spec < 0 || context.max_spec > 3) return 2;
        Builder builder(registry, metadata, context);
        builder.set_audit_only(true); builder.set_instances(false);
        builder.set_sources(rad_container_sources(file));
        const int status = architecture->arch_declare(&builder, &metadata, &context);
        complete &= status == RAD_OK && builder.errors().empty();
        const Program& program = builder.program();
        for (size_t index = 1; index < program.ops.size(); ++index) {
            const auto& operation = program.ops[index];
            std::string names, encodings;
            for (const rad_weight handle : operation.weights) {
                if (!handle || handle >= program.weights.size()) continue;
                const std::string& name = program.weights[handle].name;
                if (!names.empty()) { names += ';'; encodings += ';'; }
                names += name;
                RadEncoding encoding{}; int64_t shape[RAD_MAX_RANK]{}; uint32_t dimensions = 0;
                const int found = builder.weight_encoding(name.c_str(), &encoding, shape, &dimensions);
                char text[1024]{};
                if (found == RAD_OK) rad_enc_format(&encoding, text, sizeof(text));
                encodings += found == RAD_OK ? text : "encoding_unresolved";
            }
            for (const auto& band : operation.bands) {
                const auto* device = band.dom[RAD_DOMAIN_DEVICE].row;
                const auto* host = band.dom[RAD_DOMAIN_HOST].row;
                std::printf("%d,%d,%zu,%s,%s,%s,%s,%s,%s,%s,%s,%s\n", rank, status, index,
                    csv(operation.op).c_str(), csv(operation.base.str()).c_str(), csv(band.span).c_str(),
                    csv(device ? device->plugin : "").c_str(), csv(device ? device->info->name : "").c_str(),
                    csv(host ? host->plugin : "").c_str(), csv(band.miss[RAD_DOMAIN_DEVICE]).c_str(),
                    csv(names).c_str(), csv(encodings).c_str());
            }
        }
        for (const auto& error : builder.errors()) std::fprintf(stderr, "rank %d: %s\n", rank, error.c_str());
        std::fprintf(stderr, "audit rank=%d status=%d ops=%zu weights=%zu complete=%d\n",
                     rank, status, program.ops.size() - 1, program.weights.size() - 1, complete);
        if (builder.finish() != RAD_E_STATE) return 3;
    }
    return complete ? 0 : 1;
}
