/* Muse Glimmer's reply format: the ATEM channel protocol its chat template renders.
 *
 * The template's generation prompt ends at `<|start|>assistant`, so the model opens its own
 * channels: `<to=self>` reasoning first, then the answer:
 *
 *   to=self<|message|> ...reasoning... <|eom|><|start|>assistant to=user<|message|> ...answer...
 *
 * `reasoning_close` carries the whole channel transition, so the answer's own header is consumed
 * with the reasoning block's close and the content begins at the answer's first byte. A reply
 * that skips the reasoning opens on the answer's channel directly (`to=user<|message|> ...`);
 * `content_open` names that header so it is dropped there too.
 *
 * Tool calls are the template's ATEM dialect, emitted as their own assistant turn whose channel
 * names the tool (chat_template.jinja, render_atem):
 *
 *   to=self<|message|> ...reasoning... <|eom|><|start|>assistant to=NAME<|message|>
 *   <atem:function_calls>
 *   <atem:invoke name="NAME">
 *   <atem:parameter name="k">value</atem:parameter>
 *   </atem:invoke>
 *   </atem:function_calls>
 *
 * String parameters are written raw, everything else as JSON (the template's `tojson`); the
 * section opener is the constrained-decoding trigger, and it is also the reasoning break: a call
 * turn's channel header names the tool rather than `user`, so the answer's `reasoning_close`
 * does not match there and the reasoning block runs to the section, swallowing the `<|eom|>` and
 * the header.
 */
#pragma once
#include "rad_builder.h"

namespace museglimmer {

inline const RadChatFormat* muse_chat_format() {
    static const RadChatFormat f = [] {
        RadChatFormat x{};
        x.struct_size = sizeof(x);
        x.name = "plugin:museglimmer";
        x.reasoning_open = "to=self<|message|>";
        x.reasoning_close = "<|eom|><|start|>assistant to=user<|message|>";
        static const char* const kBreaks[] = {"<atem:function_calls>"};
        x.reasoning_breaks = kBreaks;
        x.n_reasoning_breaks = 1;
        /* The model opens its own channel; the prompt carries none. */
        x.reasoning_in_prompt = RAD_CHAT_THINK_NEVER;
        x.content_open = "to=user<|message|>";
        x.section_open = "<atem:function_calls>";
        x.section_close = "</atem:function_calls>";
        x.call_open = "<atem:invoke ";
        x.name_prefix = "name=\"";
        x.name_suffix = "\">";
        x.call_end = "</atem:invoke>";
        x.arg_prefix = "<atem:parameter name=\"";
        x.arg_name_suffix = "\">";
        x.arg_value_suffix = "</atem:parameter>";
        x.string_value = RAD_CHAT_VALUE_RAW;   /* the template writes string arguments raw */
        x.other_value = RAD_CHAT_VALUE_JSON;   /* booleans, numbers, arrays, objects as tojson */
        x.value_newlines = 1;
        /* No grammar triggers: the reference deployment (vLLM's --tool-call-parser muse) runs
         * unconstrained and parses whatever the model writes. A trigger arms the lazy tool-call
         * grammar, and with the DFlash drafter the trigger fires mid-verify -- inside drafted
         * text -- at the reasoning-to-call boundary, which spins the sampler and loops the
         * channel header (measured: drafter on = the header repeats 3-4x and the reply stalls;
         * drafter off = one header, clean call). The streaming parser reads natural ATEM as
         * it is. */
        return x;
    }();
    return &f;
}

}  /* namespace museglimmer */