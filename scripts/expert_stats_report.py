#!/usr/bin/env python3
"""expert_stats_report.py FILE.json [--kinds prefill,decode,...] [--json OUT] [--top N]

Reads the file `radiance --expert-stats FILE` writes (format radiance-expert-stats/1) and prints, per routed layer:
  the share of selections taken by the top 5 / 10 / 20 % of experts, the Gini coefficient, the experts never selected,
  and decode vs prefill: each one's top-20 % share, the share of decode selections landing on prefill's top-20 %
  experts, the Jaccard overlap of the two top-20 % sets, and the total-variation distance between the two.
Then a combined skew verdict (what PLAN-experts.md E3/E4/E5 need to decide on).

The counts are model-wide expert ids (the engine merges the ranks' slices). `--kinds` picks what "all" sums (default
prefill,decode,mixed; the draft head is reported apart). The self-check line compares each layer's selections with
its observed tokens x top_k (exact unless the drain lost histograms).
"""
import argparse
import json
import math
import signal
import sys

KINDS = ("prefill", "decode", "mixed", "draft")


def safe(s):
    """A string for display. A stats file can come from someone else's run, and its model,
    arch and map strings reach the operator's terminal: ESC there starts an OSC sequence that
    could retitle the window or worse, so control characters print as '?' and never raw."""
    return "".join("?" if (c < " " or c == "\x7f" or 0x80 <= ord(c) <= 0x9F) else c for c in str(s))


def gini(xs):
    """Gini coefficient of non-negative counts: 0 flat, (n-1)/n when one expert takes everything."""
    n = len(xs)
    s = sum(xs)
    if n == 0 or s == 0:
        return 0.0
    v = sorted(xs)
    cum = 0.0
    for i, x in enumerate(v, 1):
        cum += i * x
    return (2.0 * cum) / (n * s) - (n + 1.0) / n


def top_share(xs, frac):
    """Share of all selections taken by the top ceil(frac * n) experts."""
    n = len(xs)
    s = sum(xs)
    if n == 0 or s == 0:
        return float("nan")
    k = max(1, int(math.ceil(frac * n)))
    return sum(sorted(xs, reverse=True)[:k]) / s


def top_set(xs, frac):
    n = len(xs)
    k = max(1, int(math.ceil(frac * n)))
    order = sorted(range(n), key=lambda e: (-xs[e], e))
    return set(e for e in order[:k] if xs[e] > 0)


def tvd(a, b):
    sa, sb = sum(a), sum(b)
    if sa == 0 or sb == 0:
        return float("nan")
    return 0.5 * sum(abs(x / sa - y / sb) for x, y in zip(a, b))


def summed(counts, kinds, layer, n):
    row = [0] * n
    for k in kinds:
        for e, c in enumerate(counts.get(k, [[]] * (layer + 1))[layer]):
            row[e] += c
    return row


def fmt_pct(x):
    return "   -  " if x != x else f"{100 * x:5.1f}%"


def main():
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)   # quiet under | head
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file")
    ap.add_argument("--kinds", default="prefill,decode,mixed", help="what the 'all' columns sum")
    ap.add_argument("--json", help="also write the per-layer metrics and the verdict here")
    ap.add_argument("--top", type=int, default=0, help="also list each layer's N hottest experts (all kinds)")
    a = ap.parse_args()

    d = json.load(open(a.file))
    if d.get("format") != "radiance-expert-stats/1":
        sys.exit(f"{a.file}: not a radiance-expert-stats/1 file (format {d.get('format')!r})")
    kinds = [k for k in a.kinds.split(",") if k]
    for k in kinds:
        if k not in KINDS:
            sys.exit(f"unknown kind {k!r}; one of {', '.join(KINDS)}")
    counts = d["counts"]
    L = d["n_layers"]
    n_trunk = d.get("n_trunk_layers", L)
    lex = d["layer_experts"]
    topk = d["top_k"]

    up = d.get("uptime_s", 0.0)
    print(f"{a.file}: {safe(d.get('model', '?'))} ({safe(d.get('arch', '?'))}), TP{d.get('tp', '?')}, "
          f"{'final' if d.get('final') else 'periodic'} write after {up / 60:.1f} min")
    tok = d.get("tokens", {})
    stp = d.get("steps", {})
    print("  issued: " + ", ".join(f"{k} {tok.get(k, 0)} tokens / {stp.get(k, 0)} {'passes' if k == 'draft' else 'steps'}"
                                   for k in KINDS) + f" (mixed: {d.get('mixed_decode_tokens', 0)} decode tokens)")
    ranks = d.get("ranks", {})
    print(f"  drain: {sum(ranks.get('credited', []))} histograms credited, {sum(ranks.get('dropped', []))} lost by the "
          f"ring, {sum(ranks.get('unmapped', []))} unmapped selections, {sum(ranks.get('unclassified', []))} unclassified")
    maps = sorted(set(safe(m) for m, n in zip(d.get("map", []), lex) if n > 0))
    print(f"  ids: {', '.join(maps)}; summing {'+'.join(kinds)} as 'all'")

    # Self-check: selections == observed tokens x top_k, per kind and layer.
    bad = []
    for k in KINDS:
        sel = d["selections"][k]
        ot = d["observed_tokens"][k]
        for l in range(L):
            if lex[l] == 0 or (sel[l] == 0 and ot[l] == 0):
                continue
            if sel[l] != ot[l] * topk[l]:
                bad.append(f"{k} L{l}: {sel[l]} selections vs {ot[l]} tokens x top-{topk[l]}")
    print("  self-check: " + ("every layer's selections == observed tokens x top_k" if not bad else
                              f"{len(bad)} mismatches, e.g. " + "; ".join(bad[:4])))

    print()
    hdr = (f"{'layer':>5} {'n':>4} {'k':>3} {'sel(all)':>11} {'top5%':>6} {'top10%':>6} {'top20%':>6} {'gini':>5} "
           f"{'never':>5} | {'dec top20':>9} {'pf top20':>8} {'d@pfTop':>7} {'jacc':>5} {'tvd':>5} {'dec never':>9}")
    print(hdr)
    rows = []
    for l in range(L):
        n = lex[l]
        if n == 0:
            continue
        head = l >= n_trunk
        allk = ["draft"] if head else kinds
        x = summed(counts, allk, l, n)
        dec = summed(counts, ["decode"], l, n)
        pf = summed(counts, ["prefill"], l, n)
        r = {"layer": l, "head": head, "n": n, "top_k": topk[l], "selections": sum(x),
             "top5": top_share(x, 0.05), "top10": top_share(x, 0.10), "top20": top_share(x, 0.20),
             "gini": gini(x), "never": sum(1 for c in x if c == 0),
             "dec_top20": top_share(dec, 0.20), "pf_top20": top_share(pf, 0.20),
             "dec_never": sum(1 for c in dec if c == 0) if sum(dec) else None}
        if sum(dec) and sum(pf):
            pt = top_set(pf, 0.20)
            dt = top_set(dec, 0.20)
            r["dec_on_pf_top20"] = sum(dec[e] for e in pt) / sum(dec)
            r["jaccard_top20"] = len(pt & dt) / max(1, len(pt | dt))
            r["tvd"] = tvd(dec, pf)
        else:
            r["dec_on_pf_top20"] = r["jaccard_top20"] = r["tvd"] = float("nan")
        rows.append(r)
        print(f"{('H' if head else 'L') + str(l):>5} {n:>4} {topk[l]:>3} {r['selections']:>11} {fmt_pct(r['top5'])} "
              f"{fmt_pct(r['top10'])} {fmt_pct(r['top20'])} {r['gini']:5.3f} {r['never']:>5} | "
              f"{fmt_pct(r['dec_top20']):>9} {fmt_pct(r['pf_top20']):>8} {fmt_pct(r['dec_on_pf_top20']):>7} "
              f"{r['jaccard_top20']:5.2f} {r['tvd']:5.3f} {('-' if r['dec_never'] is None else r['dec_never']):>9}")
        if a.top:
            hot = sorted(range(n), key=lambda e: -x[e])[:a.top]
            print("        hottest: " + " ".join(f"e{e}:{x[e] / max(1, sum(x)) * 100:.1f}%" for e in hot))

    trunk = [r for r in rows if not r["head"] and r["selections"] > 0]
    verdict = {"layers": len(trunk)}
    print()
    if not trunk:
        print("verdict: no trunk selections recorded")
    else:
        def mean(key):
            v = [r[key] for r in trunk if r[key] == r[key]]
            return sum(v) / len(v) if v else float("nan")
        m20, m10, mg = mean("top20"), mean("top10"), mean("gini")
        never = sum(r["never"] for r in trunk)
        experts = sum(r["n"] for r in trunk)
        mdec = mean("dec_on_pf_top20")
        mtvd = mean("tvd")
        mdt, mpt = mean("dec_top20"), mean("pf_top20")
        # How much of decode's concentration sits on prefill's hot set: (share on prefill's top 20 % - 20 %) over
        # (decode's own top-20 % share - 20 %); 1 = the same hot experts, 0 = no better than chance.
        shr = [(r["dec_on_pf_top20"] - 0.2) / (r["dec_top20"] - 0.2) for r in trunk
               if r["dec_on_pf_top20"] == r["dec_on_pf_top20"] and r["dec_top20"] - 0.2 > 0.05]
        shared = sum(shr) / len(shr) if shr else float("nan")
        # Uniform routing puts 20 % of selections on the top 20 % (plus sampling noise); thresholds are judgement
        # calls for the E3/E4/E5 decision, stated so they can be argued with.
        if m20 >= 0.50 or mg >= 0.45:
            skew = "SKEWED"
            act = ("worth per-expert types (E4: hot/cold tables) and REAP-style pruning of the cold tail (E5); "
                   "per-layer reallocation (E3) too")
        elif m20 >= 0.35 or mg >= 0.25:
            skew = "MODERATE"
            act = "per-layer reallocation (E3) first; E4/E5 only for the most skewed layers"
        else:
            skew = "FLAT"
            act = "usage is near uniform: per-layer reallocation (E3) only; pruning would cut used experts"
        if mdec != mdec:
            dp = "no decode/prefill comparison (one of them has no selections)"
        elif mdt < 0.3 and mpt < 0.3:
            dp = "decode and prefill are both near flat, so there is no hot set to share"
        elif shared != shared:
            dp = "decode is near flat; prefill is not"
        elif shared >= 0.6:
            dp = f"decode's hot experts are mostly prefill's (shared {shared:.2f})"
        elif shared < 0.3:
            dp = f"decode and prefill favour different experts (shared {shared:.2f})"
        else:
            dp = f"decode and prefill partly share their hot experts (shared {shared:.2f})"
        verdict.update({"skew": skew, "mean_top10": m10, "mean_top20": m20, "mean_gini": mg,
                        "never_selected": never, "experts": experts, "mean_decode_on_prefill_top20": mdec,
                        "mean_tvd": mtvd, "mean_decode_top20": mdt, "mean_prefill_top20": mpt,
                        "hot_set_shared": shared, "decode_vs_prefill": dp, "action": act})
        sel = sum(r["selections"] for r in trunk)
        print(f"verdict: {skew} -- over {len(trunk)} trunk layers ({sel} selections): top 10 % of experts take "
              f"{fmt_pct(m10).strip()}, top 20 % take {fmt_pct(m20).strip()} (uniform: 20 %), mean Gini {mg:.3f}; "
              f"{never} of {experts} experts never selected")
        print(f"         {dp}: top 20 % take {fmt_pct(mdt).strip()} of decode, {fmt_pct(mpt).strip()} of prefill; "
              f"decode puts {fmt_pct(mdec).strip()} of its selections on prefill's top 20 %; mean TVD {mtvd:.3f}")
        print(f"         -> {act}")
        if sel < 100 * experts:
            print(f"         caution: {sel / experts:.0f} selections an expert on average; rank tails are noisy below ~100")
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"layers": rows, "verdict": verdict}, f, indent=1, default=lambda o: None)


if __name__ == "__main__":
    try:
        main()
    except (TypeError, ValueError, KeyError, IndexError, AttributeError) as e:
        # A file that passes the format check but has the wrong shape underneath is untrusted
        # data too: name the fault on one line rather than print a traceback to read past.
        sys.exit(f"expert_stats_report: malformed stats file: {e}")
