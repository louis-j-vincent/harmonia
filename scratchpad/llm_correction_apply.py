"""LLM-correction experiment — STAGE 2: song-level QUALITY correction via LLM.

For each held-out song, batch its WHOLE predicted (root, quality) sequence +
estimated key/mode into ONE `claude -p` call (economical; also the design intent
— song-structural context). The LLM proposes QUALITY-only corrections keyed on
key/mode consistency + repeated-progression consistency. Roots are NEVER changed
(scope = quality/chord-type; root is a separately-confirmed dead axis). Applies
corrections, then scores balanced acc + dom recall BEFORE vs AFTER, and audits
where the correction HELPED vs HURT (esp. over-correction toward diatonic).

Cost note: ~1 call/song. Uses the `claude` CLI (the only LLM access here).
"""
from __future__ import annotations
import json, subprocess, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
from train_real_audio_final import QUALITIES
DOM = QUALITIES.index("dom")
QSET = set(QUALITIES)

PROMPT = """You are a jazz/pop harmony analyst correcting an audio chord-recognition model's CHORD-QUALITY predictions for one song. The model already fixed each chord's ROOT; your ONLY job is to fix the QUALITY (chord type) where song-level context makes the current label very likely wrong.

Quality vocabulary (exactly these 7 tokens): maj, min, dom, hdim, dim, aug, sus.
- maj = major triad / maj6 / maj7 family; min = minor triad/min7 family;
- dom = dominant 7th (a major triad that functions as a V7, often resolving down a 5th);
- hdim = half-diminished (m7b5); dim = diminished; aug = augmented; sus = suspended.

Song key: {key} {mode}.
Predicted chord sequence (index : ROOT quality), in time order:
{seq}

Correction principles (apply CONSERVATIVELY — only high-confidence flips):
1. Repeated-progression consistency: if the same progression/section recurs and one instance has an outlier quality that breaks an otherwise-identical pattern, the outlier is a likely error — align it to the repeated form.
2. Key/function fit: a quality that contradicts the chord's diatonic function in {key} {mode}, while its neighbours are consistent, is suspect (e.g. the V of the key is usually dom, not maj; the ii in major is usually min; a lone dom in a run of otherwise identical maj chords in a diatonic context is suspect).
3. DO NOT over-correct toward the most common/diatonic chord. Genuine chromatic/borrowed/secondary-dominant chords are real music — only flip when the STRUCTURE (a repeat) or a clear local functional contradiction supports it, not merely because a chord is non-diatonic.
4. Never change a root. Never invent indices. Only emit flips you are confident improve accuracy; emitting NO corrections is a valid, good answer.

Respond with ONLY this JSON (no prose, no markdown fence):
{{"corrections": [{{"i": <index>, "from": "<current_quality>", "to": "<new_quality>", "reason": "<short>"}}]}}
"""


def call_llm(key, mode, seq):
    lines = "\n".join(f"{s['i']:3d} : {s['root']:<2} {s['q']}" for s in seq)
    prompt = PROMPT.format(key=key, mode=mode, seq=lines)
    r = subprocess.run(
        ["claude", "-p", "--output-format", "json", "--max-turns", "1",
         "--strict-mcp-config", "--disallowed-tools", "Bash Read Write Edit WebSearch WebFetch",
         "--system-prompt", "You are an expert jazz/pop harmony analyst. You output ONLY strict JSON, no prose."],
        input=prompt, capture_output=True, text=True, timeout=240)
    if r.returncode != 0:
        print("  [warn] claude returned", r.returncode, r.stderr[:200]); return []
    try:
        outer = json.loads(r.stdout)
        txt = outer["result"].strip()
    except Exception as e:
        print("  [warn] outer parse fail", e, r.stdout[:150]); return []
    if txt.startswith("```"):
        txt = txt.strip("`"); txt = txt[txt.find("{"):txt.rfind("}")+1]
    try:
        return json.loads(txt).get("corrections", [])
    except Exception:
        s, e = txt.find("{"), txt.rfind("}")
        try: return json.loads(txt[s:e+1]).get("corrections", [])
        except Exception as e2:
            print("  [warn] inner parse fail", e2, txt[:150]); return []


def bal_dom(preds, y, n=7):
    rec = {c: (float((preds[y == c] == c).mean()) if (y == c).sum() else 0.0) for c in range(n)}
    return float(np.mean([rec[c] for c in range(n)])), rec[DOM], rec


def main():
    songs = json.loads((REPO / "scratchpad/llm_correction_songs.json").read_text())
    base = np.load(REPO / "scratchpad/llm_correction_base.npz", allow_pickle=True)
    sid = base["song_id"]; t0 = base["t0"]
    pred_q = base["pred_q"].copy(); gt_q = base["gt_q"]; pred_root = base["pred_root"]

    after = pred_q.copy()
    audit = []  # (song, i, from, to, gt, correct_flip?)
    n_calls = 0
    for s, info in songs.items():
        m = np.where(sid == s)[0]
        order = m[np.argsort(t0[m])]           # global idx in time order; seq['i']==position
        corrs = call_llm(info["key"], info["mode"], info["seq"])
        n_calls += 1
        for c in corrs:
            try:
                i = int(c["i"]); to = c["to"]
            except Exception:
                continue
            if to not in QSET or i < 0 or i >= len(order):
                continue
            g = int(order[i])
            frm = pred_q[g]
            after[g] = QUALITIES.index(to)
            gtc = int(gt_q[g])
            # correct flip = moved onto GT; harmful = moved off GT
            helped = (frm != gtc) and (after[g] == gtc)
            hurt = (frm == gtc) and (after[g] != gtc)
            audit.append((s, i, QUALITIES[frm], to, QUALITIES[gtc], helped, hurt))
        print(f"  {s}: {len(corrs)} corrections", flush=True)

    b_bal, b_dom, b_rec = bal_dom(pred_q, gt_q)
    a_bal, a_dom, a_rec = bal_dom(after, gt_q)
    n_flips = len(audit)
    helped = sum(a[5] for a in audit); hurt = sum(a[6] for a in audit)
    neutral = n_flips - helped - hurt

    print("\n" + "=" * 62)
    print(f"LLM song-level QUALITY correction — {n_calls} songs, {n_flips} flips")
    print(f"  helped (wrong->right): {helped}   hurt (right->wrong): {hurt}   neutral(wrong->wrong): {neutral}")
    print(f"\n              BEFORE     AFTER      delta")
    print(f"  balanced   {b_bal:.3f}      {a_bal:.3f}     {a_bal-b_bal:+.3f}")
    print(f"  dom recall {b_dom:.3f}      {a_dom:.3f}     {a_dom-b_dom:+.3f}")
    print(f"  raw acc    {(pred_q==gt_q).mean():.3f}      {(after==gt_q).mean():.3f}     {(after==gt_q).mean()-(pred_q==gt_q).mean():+.3f}")
    print("\n  per-class recall before/after:")
    for c in range(7):
        print(f"    {QUALITIES[c]:5s}  {b_rec[c]:.3f} -> {a_rec[c]:.3f}")
    print("\n  HARMFUL flips (right->wrong):")
    for a in audit:
        if a[6]:
            print(f"    {a[0]} i={a[1]}: {a[2]}->{a[3]} (GT={a[4]})")

    np.savez(REPO / "scratchpad/llm_correction_after.npz", after=after,
             pred_q=pred_q, gt_q=gt_q, song_id=sid)
    json.dump([list(map(lambda x: x.item() if hasattr(x,'item') else x, a)) for a in audit],
              open(REPO / "scratchpad/llm_correction_audit.json", "w"))


if __name__ == "__main__":
    main()
