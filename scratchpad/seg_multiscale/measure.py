"""Score every granularity arm on the 50 canonical charts. Metric: METRIC.md."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from harness import all_charts                                     # noqa: E402
from harmonia.models.section_vocab import form_string              # noqa: E402
from harmonia.output.chart_display import _fold_units, _group_to_min_bars  # noqa: E402
from anchor import anchor_sections                                # noqa: E402
from variants import vocab_variant                                 # noqa: E402

ARMS = ("auto", "fix2", "fix4", "fix8", "hier", "coarse1st",
        "anchor3", "anchor4", "anchor6", "anchor8",
        "anchorS4", "anchorN4",   # ablations: sharp-only zones / no zones at all
        "anchorG4", "anchorG6")   # + global-period prior from the lag profile


def metrics(vocab, n_bars, min_bars):
    if not vocab:
        return None
    units = _group_to_min_bars(vocab, min_bars=min_bars)
    order, groups, _sfx = _fold_units(units)
    m1 = len(order)
    m2_ok = sum(1 for s in vocab if (s["bar1"] - s["bar0"]) % max(1, s["d_bars"]) == 0)
    m2 = m2_ok / len(vocab)
    letters = Counter()
    for s in vocab:
        letters[s["label"]] += s["bar1"] - s["bar0"]
    occ = Counter(s["label"] for s in vocab)
    # "recurs" must count MUSICAL occurrences, not play-order runs. `A×3` is one
    # run but three plays, and an arm that merges adjacent repeats into one run
    # (the anchor arm does) was being scored as if its dominant section never
    # came back — a defect in the metric, not in the arm. Fixed after the first
    # sweep; both readings are reported so the change is auditable.
    plays = Counter()
    for s in vocab:
        plays[s["label"]] += max(1, s["reps"])
    dom, dom_bars = letters.most_common(1)[0]
    m3 = (len(letters) <= 5 and dom_bars / n_bars >= 0.30 and plays[dom] >= 2)
    m3_runs = (len(letters) <= 5 and dom_bars / n_bars >= 0.30 and occ[dom] >= 2)
    rec = {lab for lab in occ if plays[lab] >= 2}
    m4 = sum(letters[l] for l in rec) / n_bars
    m5 = sum(1 for s in vocab if (s["bar1"] - s["bar0"]) < 4)
    return {"n_written": m1, "len_honest": m2, "sane": bool(m3),
            "sane_runs": bool(m3_runs), "rec_cov": m4,
            "frag": m5, "n_letters": len(letters), "form": form_string(vocab)}


def main(min_bars: int = 4):
    charts = all_charts()
    rows = []
    for c in charts:
        r = {"slug": c["slug"], "file": c["file"].name, "n_bars": c["n_bars"]}
        for arm in ARMS:
            try:
                if arm.startswith("anchor"):
                    tail = arm[6:]
                    zf = {"S": "sharp", "N": "none"}.get(tail[0], "blur")
                    sig = float(tail[1:] if tail[0] in "SNG" else tail)
                    v = anchor_sections(c["bars"], c["n_bars"], tonic_pc=c["tonic_pc"],
                                        bpb=c["bpb"], sigma_bars=sig, zones_from=zf,
                                        global_period=tail[0] == "G")
                else:
                    v = vocab_variant(c["bars"], c["n_bars"], tonic_pc=c["tonic_pc"],
                                      bpb=c["bpb"], arm=arm)
                r[arm] = metrics(v, c["n_bars"], min_bars)
            except Exception as e:                      # noqa: BLE001
                r[arm] = {"error": f"{type(e).__name__}: {e}"}
        rows.append(r)

    out = HERE / f"sweep_min{min_bars}.json"
    out.write_text(json.dumps(rows, indent=1))

    def _fired(r, arm):
        return bool(r.get(arm)) and "error" not in r[arm]

    common = [r for r in rows if all(_fired(r, a) for a in ARMS)]

    def table(subset, title):
        print(f"\n{title}  (n={len(subset)})")
        hdr = (f"{'arm':9s} {'fired':>7s} {'M1 sect':>8s} {'M2 hon':>7s} "
               f"{'M3 sane':>8s} {'M4 cov':>7s} {'M5 frag':>8s} {'letters':>8s}")
        print(hdr)
        print("-" * len(hdr))
        for arm in ARMS:
            ok = [r[arm] for r in subset if _fired(r, arm)]
            if not ok:
                print(f"{arm:9s}  none")
                continue
            n = len(ok)
            print(f"{arm:9s} {n:4d}/{len(subset):<2d} "
                  f"{sum(m['n_written'] for m in ok)/n:8.2f} "
                  f"{sum(m['len_honest'] for m in ok)/n:7.3f} "
                  f"{sum(m['sane'] for m in ok)/n:7.1%} "
                  f"{sum(m['rec_cov'] for m in ok)/n:7.3f} "
                  f"{sum(m['frag'] for m in ok)/n:8.2f} "
                  f"{sum(m['n_letters'] for m in ok)/n:8.2f}")

    print(f"corpus: {len(rows)} charts   HARMONIA_MIN_SECTION_BARS={min_bars}")
    table(rows, "ALL CHARTS — each arm on the charts IT fires on (not comparable)")
    table(common, "COMMON SUBSET — every arm fires (the fair comparison)")
    errs = Counter(r[arm].get("error", "")[:60] for r in rows for arm in ARMS
                   if r.get(arm) and "error" in r[arm])
    for e, k in errs.items():
        print(f"  ERROR x{k}: {e}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 4)
