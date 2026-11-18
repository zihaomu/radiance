int hint_reference(const RadArgs* args, RadStream) {
    auto input = [&](int operand) { return static_cast<const int32_t*>(args->t[operand].data); };
    const int64_t heads = rad_args_geti_or(args, "heads", 0), order = rad_args_geti_or(args, "ngram", 0), eos = rad_args_geti_or(args, "eos", 0);
    const bool history = rad_args_geti_or(args, "hist", 0) != 0;
    const int64_t context = order - 1, sequences = rad_tensor_numel(args->t + 2) - 1, slots = rad_tensor_numel(args->t + 7) / (context * 2);
    auto* tail = static_cast<int32_t*>(args->t[7].data); auto* output = static_cast<int32_t*>(args->t[8].data);
    const auto *multipliers = static_cast<const int64_t*>(args->t[3].data), *sizes = static_cast<const int64_t*>(args->t[4].data), *offsets = static_cast<const int64_t*>(args->t[5].data);
    for (int64_t sequence = 0; sequence < sequences; ++sequence) {
        const int64_t begin = input(2)[sequence], end = input(2)[sequence + 1], slot = input(6) ? input(6)[sequence] : sequence;
        auto* ids = output + sequence * args->t[8].stride[0];
        if (end <= begin || slot < 0) { for (int64_t head = 0; head < heads; ++head) ids[head] = int32_t(offsets[head]); continue; }
        int64_t first = end - 1;
        if (history) while (first > begin && input(1)[first - 1] == input(1)[end - 1]) --first;
        const int64_t base = slot % slots * context * 2;
        std::vector<int64_t> previous(size_t(context), eos);
        for (int64_t shift = 1; shift <= context; ++shift) {
            if (!history) previous[size_t(shift - 1)] = tail[base + context * 2 - shift];
            else if (first - shift >= begin) previous[size_t(shift - 1)] = input(0)[first - shift];
            else previous[size_t(shift - 1)] = tail[base + context - (shift - (first - begin))];
        }
        std::vector<int32_t> next(size_t(context), input(0)[end - 1]);
        for (int64_t index = 0; index + 1 < context; ++index) next[size_t(index)] = int32_t(previous[size_t(context - index - 2)]);
        for (int64_t index = 0; index < context; ++index) { tail[base + context + index] = next[size_t(index)]; if (history) tail[base + index] = next[size_t(index)]; }
        std::vector<int64_t> shifted{input(0)[end - 1]};
        for (int64_t index = 0; index < context; ++index) shifted.push_back(shifted.back() == eos ? eos : previous[size_t(index)]);
        for (int64_t head = 0; head < heads; ++head) {
            const int64_t terms = head / (heads / context) + 2; uint64_t bits = 0;
            for (int64_t term = 0; term < terms; ++term) bits ^= uint64_t(shifted[size_t(term)]) * uint64_t(multipliers[term]);
            int64_t signed_bits; std::memcpy(&signed_bits, &bits, sizeof(bits));
            int64_t remainder = signed_bits % sizes[head]; if (remainder < 0) remainder += sizes[head];
            ids[head] = int32_t(remainder + offsets[head]);
        }
    }
    return RAD_OK;
}
void ngram_case(int64_t count) {
    const int64_t heads = 16, order = 3, window = count + 4, eos = 17;
    Blob tokens(count + 2, RAD_I32), state(3 * window, RAD_I32), cu(3, RAD_I32), mult(order, RAD_I64), sizes(heads, RAD_I64), offsets(heads, RAD_I64);
    Blob indices(2, RAD_I32), initialized(2, RAD_I32), accepted(2, RAD_I32), ids(count * (heads + 3), RAD_I32), ahead(2 * (heads + 3), RAD_I32);
    cu.data<int32_t>()[0] = 0; cu.data<int32_t>()[1] = int32_t(count / 2); cu.data<int32_t>()[2] = int32_t(count);
    mult.data<int64_t>()[0] = 24000000000001ll; mult.data<int64_t>()[1] = -51000000000007ll; mult.data<int64_t>()[2] = 0x7ffffffffffffffdll;
    indices.data<int32_t>()[0] = 2; indices.data<int32_t>()[1] = 0;
    for (int64_t head = 0; head < heads; ++head) { sizes.data<int64_t>()[head] = 101 + head * 2; offsets.data<int64_t>()[head] = head * 200; }
    for (int variant = 0; variant < 4; ++variant) {
        for (size_t index = 0; index < tokens.size(); ++index) tokens.data<int32_t>()[index] = index % 4 == 1 ? int32_t(eos) : int32_t(1 + (index + variant) % 29);
        for (size_t index = 0; index < state.size(); ++index) state.data<int32_t>()[index] = int32_t(index % 31);
        initialized.data<int32_t>()[0] = variant & 1; initialized.data<int32_t>()[1] = 1;
        accepted.data<int32_t>()[0] = 2; accepted.data<int32_t>()[1] = 1;
        run("ngram_ids state_optional_eos_overflow_ahead", r3dh_mtp_ple_rad_ngram_ids, ref_ngram_ids,
            {RAD_INT("M", 32), RAD_INT("heads", heads), RAD_INT("ngram", order), RAD_INT("eos", eos)},
            {tensor(tokens, {count + ((variant & 2) ? 2 : 0)}), tensor(state, {3, window}), tensor(cu, {3}), tensor(mult, {order}), tensor(sizes, {heads}), tensor(offsets, {heads}),
             variant & 1 ? tensor(indices, {2}) : RadTensor{}, variant & 2 ? tensor(initialized, {2}) : RadTensor{}, variant & 1 ? tensor(accepted, {2}) : RadTensor{},
             tensor(ids, {count, heads}, heads + 3), variant & 2 ? tensor(ahead, {2, heads}, heads + 3) : RadTensor{}});
    }
    Blob positions(count + 2, RAD_I32), tail(3 * 2 * (order - 1), RAD_I32), hints(2 * (heads + 3), RAD_I32);
    for (size_t index = 0; index < tail.size(); ++index) tail.data<int32_t>()[index] = int32_t(3 + index % 19);
    for (size_t index = 0; index < positions.size(); ++index) positions.data<int32_t>()[index] = int32_t(index / 2);
    for (int history : {1, 0, 0}) run("ngram_hint committed_chain_repeated_position", r3dh_mtp_ple_ngram_hint, hint_reference,
        {RAD_INT("M", 32), RAD_INT("heads", heads), RAD_INT("ngram", order), RAD_INT("eos", eos), RAD_INT("hist", history)},
        {tensor(tokens, {count + 2}), tensor(positions, {count + 2}), tensor(cu, {3}), tensor(mult, {order}), tensor(sizes, {heads}), tensor(offsets, {heads}),
         history ? tensor(indices, {2}) : RadTensor{}, tensor(tail, {3, 2 * (order - 1)}), tensor(hints, {2, heads}, heads + 3)});
}
void select_case(int64_t count) {
    const int64_t width = 2560, vocabulary = 19, shortlist = 8, offset = 500;
    const int64_t row_bytes = (width + 2 * (width / 128) + 15) & ~int64_t(15);
    Blob input(count * width, RAD_BF16), packed(vocabulary * row_bytes, RAD_F8E4M3), decoded(vocabulary * width, RAD_F32);
    Blob candidates(count * shortlist, RAD_I32), values(count * shortlist, RAD_F32), pairs(count * shortlist * 2, RAD_F32);
    fill(input, 3);
    for (int format = 0; format < 2; ++format) {
        for (int64_t row = 0; row < vocabulary; ++row) {
            auto* stored = packed.data<uint8_t>() + row * row_bytes;
            for (int64_t column = 0; column < width; ++column) stored[column] = row < 2 ? 0 : uint8_t((column * 13 + row * 7) % (format ? 256 : 111));
            for (int64_t group = 0; group < width / 128; ++group) reinterpret_cast<uint16_t*>(stored + width)[group] = rad_f32_to_bf16(0.001013f * float(1 + group % 7));
            for (int64_t column = 0; column < width; ++column) decoded.data<float>()[row * width + column] =
                rad_load_f32(stored, format ? RAD_I8 : RAD_F8E4M3, column) * rad_bf16_to_f32(reinterpret_cast<uint16_t*>(stored + width)[column / 128]);
        }
        for (int optional = 0; optional < 2; ++optional) {
            for (int64_t row = 0; row < count; ++row) for (int64_t candidate = 0; candidate < shortlist; ++candidate)
                candidates.data<int32_t>()[row * shortlist + candidate] = candidate == 7 ? -1 : (candidate == 6 ? int32_t(offset - 1) : int32_t(offset + (candidate < 2 ? 1 - candidate : (candidate + row) % vocabulary)));
            auto weight = tensor(packed, {vocabulary, row_bytes}); weight.dtype = format ? RAD_I8 : RAD_F8E4M3;
            std::vector<RadTensor> operands{tensor(input, {count, width}), weight, tensor(candidates, {count, shortlist}), tensor(candidates, {count, shortlist}), tensor(values, {count, shortlist}), optional ? tensor(pairs, {count, shortlist * 2}) : RadTensor{}};
            auto oracle = operands; oracle[1] = tensor(decoded, {vocabulary, width});
            run("logit_rerank packed_head_inplace_optional_pairs", r3dh_mtp_ple_logit_rerank, ref_logit_rerank,
                {RAD_INT("M", 32), RAD_INT("n_embd", width), RAD_INT("n_vocab", vocabulary), RAD_INT("vocab_off", offset), RAD_INT("R", shortlist)}, operands, oracle);
        }
    }
    Blob gathered(count * 3 * shortlist * 2, RAD_F32), indices(count * shortlist, RAD_I32);
    for (int64_t world : {2, 3}) {
        for (int64_t row = 0; row < count; ++row) for (int64_t candidate = 0; candidate < world * shortlist; ++candidate) {
            const int32_t id = candidate % 5 == 0 ? -1 : int32_t((candidate * 7) % 11);
            std::memcpy(gathered.data<float>() + (row * world * shortlist + candidate) * 2, &id, sizeof(id));
            gathered.data<float>()[(row * world * shortlist + candidate) * 2 + 1] = float(candidate % 3);
        }
        run("row_topk_merge world_ties_duplicate_invalid", r3dh_mtp_ple_rowtopkmerge, ref_row_topk_merge,
            {RAD_INT("M", 32), RAD_INT("R", shortlist), RAD_INT("world_size", world)},
            {tensor(gathered, {count, world * shortlist * 2}), tensor(indices, {count, shortlist}), tensor(values, {count, shortlist})});
    }
}
