"""Section-structure stats over the iReal Pro "Pop 400" corpus.

Characterizes (1) section ARCHITECTURE (occurrence lengths, within-tune
same-label length consistency, tail behavior) and (2) the LANGUAGE of
section label sequences (occurrence-level grammar, transitions), to give
harmonia's section-inference stage empirical priors.

Run: .venv/bin/python scratchpad/pop400_section_grammar.py
(repo root; PYTHONPATH=. not needed if run via .venv/bin/python from root
since harmonia/ is importable as a package from cwd)

Writes nothing; prints all tables. docs/ireal_pop400_section_grammar.md is
the write-up built from this output.
"""

from __future__ import annotations

import re
import urllib.parse
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

from pyRealParser import Tune

from harmonia.data.ireal_corpus import load_playlist, sectionized_measures

CORPUS_PATH = Path("data/ireal/pop400.txt")

# ─────────────────────────────────────────────────────────────────────────
# 0. Coverage check
# ─────────────────────────────────────────────────────────────────────────


def coverage_check() -> None:
    print("=" * 78)
    print("0. COVERAGE CHECK")
    print("=" * 78)
    text = CORPUS_PATH.read_text()
    urls = re.findall(r'irealb(?:ook)?://[^\s"<>]+', text)
    print(f"irealb:// URLs found in file: {len(urls)}")
    url = urllib.parse.unquote(urls[0])
    m = re.match(r'irealb://([^"]+)', url)
    songs = re.split("===", m.group(1))
    empty = sum(1 for s in songs if s == "")
    nonempty = len(songs) - empty
    print(f"raw '===' segments in the single URL: {len(songs)} "
          f"({empty} empty, {nonempty} non-empty)")

    ok, failed = 0, []
    for s in songs:
        if s == "":
            continue
        try:
            Tune(s)
            ok += 1
        except Exception as e:
            failed.append((s.split("=")[0], str(e)))
    print(f"parse successes: {ok}  /  failures: {len(failed)}")
    for title, err in failed:
        print(f"  FAILED: {title!r} -> {err}")
    print(
        "\nExplanation: the file is titled 'pop400' but only contains 346 "
        "song segments (not ~400) after splitting the single irealb:// URL "
        "on '==='; all 346 are non-empty. Of those, 345 parse and 1 "
        "('Tequila', The Champs) raises a hard RuntimeError in "
        "pyRealParser._fill_codas (4 'Q' coda markers where 0/1/2 are "
        "handled) and is silently dropped by load_playlist's try/except — "
        "this is the same coda edge case flagged as a 'non-blocking warning' "
        "in the task brief, but it is in fact a full parse failure for this "
        "one tune, not a warning. No other gap to explain."
    )


# ─────────────────────────────────────────────────────────────────────────
# Shared data prep
# ─────────────────────────────────────────────────────────────────────────


def occurrences(labels: list[str]) -> list[tuple[str, int, int]]:
    """Run-length encode a per-bar label sequence into (label, start, length)."""
    runs: list[tuple[str, int, int]] = []
    for lab in labels:
        if runs and runs[-1][0] == lab:
            s, st, ln = runs[-1]
            runs[-1] = (s, st, ln + 1)
        else:
            runs.append((lab, len(runs) and (runs[-1][1] + runs[-1][2]) or 0, 1))
    return runs


def load_corpus():
    tunes = load_playlist(CORPUS_PATH)
    data = []
    for t in tunes:
        sm = sectionized_measures(t)
        labels = [lab for lab, _ in sm]
        if not labels:
            continue
        occ = occurrences(labels)
        data.append(
            dict(
                title=t.title,
                time_signature=t.time_signature,
                n_bars=len(labels),
                labels=labels,
                occ=occ,  # list of (label, start_bar, length)
            )
        )
    return data


# ─────────────────────────────────────────────────────────────────────────
# 1. Corpus overview
# ─────────────────────────────────────────────────────────────────────────


def quartiles(xs: list[int]) -> tuple[float, float, float]:
    xs = sorted(xs)
    n = len(xs)
    def pct(p):
        if n == 1:
            return xs[0]
        k = p * (n - 1)
        f = int(k)
        c = min(f + 1, n - 1)
        return xs[f] + (xs[c] - xs[f]) * (k - f)
    return pct(0.25), pct(0.5), pct(0.75)


def corpus_overview(data) -> None:
    print("\n" + "=" * 78)
    print("1. CORPUS OVERVIEW")
    print("=" * 78)
    ts_counts = Counter(d["time_signature"] for d in data)
    print(f"\nTunes loaded: {len(data)}")
    print("\nTime signature distribution:")
    for ts, n in ts_counts.most_common():
        print(f"  {ts[0]}/{ts[1]:<3d} {n:4d}  ({100*n/len(data):.1f}%)")

    lengths = [d["n_bars"] for d in data]
    q1, q2, q3 = quartiles(lengths)
    print(f"\nExpanded form length (bars): min={min(lengths)} "
          f"Q1={q1:.0f} median={q2:.0f} Q3={q3:.0f} max={max(lengths)}")

    n_occ = [len(d["occ"]) for d in data]
    q1, q2, q3 = quartiles(n_occ)
    print(f"Occurrences per tune: min={min(n_occ)} Q1={q1:.0f} "
          f"median={q2:.0f} Q3={q3:.0f} max={max(n_occ)}")

    n_labels = [len(set(lab for lab, _, _ in d["occ"])) for d in data]
    lbl_counter = Counter(n_labels)
    print("\nDistinct labels per tune:")
    for k in sorted(lbl_counter):
        print(f"  {k} labels: {lbl_counter[k]:4d} tunes "
              f"({100*lbl_counter[k]/len(data):.1f}%)")

    # which labels appear at all, and how many bars/occurrences each covers
    label_bar_totals = Counter()
    label_occ_totals = Counter()
    for d in data:
        for lab, _, ln in d["occ"]:
            label_bar_totals[lab] += ln
            label_occ_totals[lab] += 1
    print("\nLabel totals across corpus (bars / occurrences):")
    for lab, nbars in label_bar_totals.most_common():
        print(f"  {lab}: {nbars:5d} bars, {label_occ_totals[lab]:4d} occurrences")


# ─────────────────────────────────────────────────────────────────────────
# 2. Occurrence lengths
# ─────────────────────────────────────────────────────────────────────────

LEN_BUCKETS = [2, 4, 8, 12, 16, 24, 32]


def occurrence_length_table(data, labels_filter=None, ts_filter=None) -> None:
    all_lens = []
    by_label = defaultdict(list)
    for d in data:
        if ts_filter is not None and d["time_signature"] != ts_filter:
            continue
        for lab, _, ln in d["occ"]:
            if labels_filter is not None and lab not in labels_filter:
                continue
            all_lens.append(ln)
            by_label[lab].append(ln)

    def row(name, lens):
        n = len(lens)
        if n == 0:
            return
        share = {b: 100 * sum(1 for x in lens if x == b) / n for b in LEN_BUCKETS}
        pct_mult4 = 100 * sum(1 for x in lens if x % 4 == 0) / n
        pct_pow2 = 100 * sum(1 for x in lens if x > 0 and (x & (x - 1)) == 0) / n
        pct_odd = 100 * sum(1 for x in lens if x % 2 == 1) / n
        med = median(lens)
        bucket_str = " ".join(f"{b}:{share[b]:4.1f}%" for b in LEN_BUCKETS)
        print(f"  {name:8s} n={n:5d} median={med:5.1f}  mult4={pct_mult4:5.1f}%  "
              f"pow2={pct_pow2:5.1f}%  odd={pct_odd:5.1f}%   [{bucket_str}]")

    row("ALL", all_lens)
    for lab in sorted(by_label, key=lambda l: -len(by_label[l])):
        row(lab, by_label[lab])


def occurrence_lengths(data) -> None:
    print("\n" + "=" * 78)
    print("2. OCCURRENCE LENGTHS (bars)")
    print("=" * 78)
    print("\nAll time signatures:")
    occurrence_length_table(data)
    print("\n4/4 only:")
    occurrence_length_table(data, ts_filter=(4, 4))


# ─────────────────────────────────────────────────────────────────────────
# 3. Within-tune same-label length consistency
# ─────────────────────────────────────────────────────────────────────────


def within_tune_consistency(data) -> None:
    print("\n" + "=" * 78)
    print("3. WITHIN-TUNE SAME-LABEL LENGTH CONSISTENCY")
    print("=" * 78)

    n_pairs_tested = 0  # (tune, label) groups with >=2 occurrences
    n_consistent = 0
    deltas = []  # (delta of deviant occurrence vs modal length)
    last_is_deviant = 0
    last_not_deviant = 0
    total_with_deviant_last = 0
    ratio_counter = Counter()  # e.g. "half", "double", "+2", "-2", "other"

    for d in data:
        by_label = defaultdict(list)
        for lab, _, ln in d["occ"]:
            by_label[lab].append(ln)
        for lab, lens in by_label.items():
            if len(lens) < 2:
                continue
            n_pairs_tested += 1
            if len(set(lens)) == 1:
                n_consistent += 1
                continue
            # inconsistent: find modal length, classify deviants
            modal = Counter(lens).most_common(1)[0][0]
            deviant_lens = [l for l in lens if l != modal]
            for l in deviant_lens:
                delta = l - modal
                deltas.append(delta)
                if l == modal * 2:
                    ratio_counter["double"] += 1
                elif modal == l * 2:
                    ratio_counter["half"] += 1
                elif delta == 2:
                    ratio_counter["+2"] += 1
                elif delta == -2:
                    ratio_counter["-2"] += 1
                elif delta == 4:
                    ratio_counter["+4"] += 1
                elif delta == -4:
                    ratio_counter["-4"] += 1
                else:
                    ratio_counter["other"] += 1
            # is the LAST occurrence of this label (within the tune) the deviant?
            occs_of_label = [(st, ln) for lb, st, ln in d["occ"] if lb == lab]
            last_len = occs_of_label[-1][1]
            total_with_deviant_last += 1
            if last_len != modal:
                last_is_deviant += 1
            else:
                last_not_deviant += 1

    print(f"\n(tune, label) groups with >=2 occurrences: {n_pairs_tested}")
    print(f"  all occurrences share one length: {n_consistent} "
          f"({100*n_consistent/n_pairs_tested:.1f}%)")
    print(f"  inconsistent (>=1 deviant length): {n_pairs_tested - n_consistent} "
          f"({100*(n_pairs_tested-n_consistent)/n_pairs_tested:.1f}%)")

    print(f"\nOf the {n_pairs_tested - n_consistent} inconsistent groups, is the "
          f"LAST occurrence the deviant one (vs the group's modal length)?")
    print(f"  last occurrence IS the/a deviant: {last_is_deviant} "
          f"({100*last_is_deviant/total_with_deviant_last:.1f}%)")
    print(f"  last occurrence is NOT deviant (deviant is earlier): {last_not_deviant} "
          f"({100*last_not_deviant/total_with_deviant_last:.1f}%)")

    print(f"\nDeviant-length delta pattern (delta = deviant_length - modal_length, "
          f"n={len(deltas)}):")
    for k, v in ratio_counter.most_common():
        print(f"  {k:8s} {v:4d}  ({100*v/len(deltas):.1f}%)")
    delta_counter = Counter(deltas)
    print("  raw delta histogram (top 10):")
    for delta, n in delta_counter.most_common(10):
        print(f"    delta={delta:+3d}: {n}")


# ─────────────────────────────────────────────────────────────────────────
# 4. Tail behavior
# ─────────────────────────────────────────────────────────────────────────


def tail_behavior(data) -> None:
    print("\n" + "=" * 78)
    print("4. TAIL BEHAVIOR (final occurrence)")
    print("=" * 78)

    final_label_counter = Counter()
    seen_before_count = 0
    new_material_count = 0
    ends_with_immediate_dup = 0
    truncated_count = 0
    extended_count = 0
    canonical_count = 0
    single_occurrence_final = 0

    n = len(data)
    for d in data:
        occ = d["occ"]
        final_lab, final_start, final_len = occ[-1]
        final_label_counter[final_lab] += 1

        earlier_labels = set(lab for lab, _, _ in occ[:-1])
        if final_lab in earlier_labels:
            seen_before_count += 1
        else:
            new_material_count += 1

        if len(occ) >= 2 and occ[-2][0] == final_lab:
            ends_with_immediate_dup += 1

        # compare final occurrence length to that label's modal (canonical)
        # length among ALL its occurrences in this tune (including final)
        same_label_lens = [ln for lab, _, ln in occ if lab == final_lab]
        if len(same_label_lens) < 2:
            single_occurrence_final += 1
            continue
        modal = Counter(same_label_lens).most_common(1)[0][0]
        if final_len == modal:
            canonical_count += 1
        elif final_len < modal:
            truncated_count += 1
        else:
            extended_count += 1

    print(f"\nTunes: {n}")
    print("\nFinal occurrence's label:")
    for lab, cnt in final_label_counter.most_common():
        print(f"  {lab}: {cnt:4d}  ({100*cnt/n:.1f}%)")

    print(f"\nFinal occurrence label already seen earlier in the tune: "
          f"{seen_before_count} ({100*seen_before_count/n:.1f}%)")
    print(f"Final occurrence is genuinely new material (label unseen before): "
          f"{new_material_count} ({100*new_material_count/n:.1f}%)")

    print(f"\nForm ends with immediate same-label duplication (... X X end): "
          f"{ends_with_immediate_dup} ({100*ends_with_immediate_dup/n:.1f}%)")

    multi = n - single_occurrence_final
    print(f"\nOf {multi} tunes where the final label recurs (>=2 occurrences of "
          f"that label in the tune):")
    print(f"  final occurrence AT canonical (modal) length: {canonical_count} "
          f"({100*canonical_count/multi:.1f}%)")
    print(f"  final occurrence TRUNCATED (shorter than modal): {truncated_count} "
          f"({100*truncated_count/multi:.1f}%)")
    print(f"  final occurrence EXTENDED (longer than modal): {extended_count} "
          f"({100*extended_count/multi:.1f}%)")
    print(f"({single_occurrence_final} tunes: final label occurs only once in "
          f"the tune -> no canonical length to compare against, excluded above)")


# ─────────────────────────────────────────────────────────────────────────
# 5. Sequence language
# ─────────────────────────────────────────────────────────────────────────


def collapse_immediate_dups(s: str) -> str:
    out = []
    for ch in s:
        if not out or out[-1] != ch:
            out.append(ch)
    return "".join(out)


def sequence_language(data) -> None:
    print("\n" + "=" * 78)
    print("5. SEQUENCE LANGUAGE")
    print("=" * 78)

    occ_strings = []
    collapsed_strings = []
    for d in data:
        s = "".join(lab for lab, _, _ in d["occ"])
        occ_strings.append(s)
        collapsed_strings.append(collapse_immediate_dups(s))

    print("\nTop 20 occurrence-level label strings (repeats NOT further collapsed):")
    for s, cnt in Counter(occ_strings).most_common(20):
        print(f"  {s:20s} {cnt:4d}  ({100*cnt/len(data):.1f}%)")

    print("\nTop 20 forms after collapsing immediate same-label duplicates "
          "(e.g. 'iAABB' -> 'iAB'):")
    for s, cnt in Counter(collapsed_strings).most_common(20):
        print(f"  {s:20s} {cnt:4d}  ({100*cnt/len(data):.1f}%)")

    # bigram transitions with START/END
    trans = defaultdict(Counter)
    for d in data:
        occ = d["occ"]
        seq = ["START"] + [lab for lab, _, _ in occ] + ["END"]
        for a, b in zip(seq, seq[1:]):
            trans[a][b] += 1

    all_states = sorted(set(trans.keys()) | {k for c in trans.values() for k in c})
    # order: START, then main labels by frequency, then END
    main_labels = [l for l in ["i", "A", "B", "C", "D"] if l in all_states]
    other_labels = sorted(l for l in all_states if l not in main_labels + ["START", "END"])
    order = ["START"] + main_labels + other_labels + ["END"]

    print("\nBigram transition COUNTS (row = from, col = to):")
    header = "        " + " ".join(f"{s:>7s}" for s in order)
    print(header)
    for a in order:
        if a == "END":
            continue
        row = " ".join(f"{trans[a].get(b, 0):7d}" for b in order)
        print(f"  {a:6s}{row}")

    print("\nBigram transition ROW-NORMALIZED % (row = from, col = to):")
    print(header)
    for a in order:
        if a == "END":
            continue
        total = sum(trans[a].values())
        if total == 0:
            continue
        row = " ".join(f"{100*trans[a].get(b, 0)/total:6.1f}%" for b in order)
        print(f"  {a:6s}{row}")

    # self-transition rate per label, overall and by position-in-form third
    print("\nSelf-transition (X -> X) rate per label:")
    for lab in main_labels:
        total = sum(trans[lab].values())
        selfr = trans[lab].get(lab, 0)
        if total:
            print(f"  {lab}: {selfr}/{total} = {100*selfr/total:.1f}%")

    print("\nSelf-transition rate by position-in-form third (first/middle/last):")
    pos_trans = {"first": defaultdict(lambda: [0, 0]),
                 "middle": defaultdict(lambda: [0, 0]),
                 "last": defaultdict(lambda: [0, 0])}
    for d in data:
        occ = d["occ"]
        n = len(occ)
        for i in range(n - 1):
            lab = occ[i][0]
            third = "first" if i < n / 3 else ("last" if i >= 2 * n / 3 else "middle")
            is_self = occ[i + 1][0] == lab
            pos_trans[third][lab][0] += int(is_self)
            pos_trans[third][lab][1] += 1
    for third in ["first", "middle", "last"]:
        print(f"  {third}:")
        for lab in main_labels:
            selfr, total = pos_trans[third].get(lab, [0, 0])
            if total:
                print(f"    {lab}: {selfr}/{total} = {100*selfr/total:.1f}%")

    # verdict-support numbers
    print("\nVerdict-support numbers:")
    n = len(data)
    b_is_final = sum(1 for d in data if d["occ"][-1][0] == "B")
    a_is_final = sum(1 for d in data if d["occ"][-1][0] == "A")
    print(f"  B is the final occurrence in {b_is_final}/{n} tunes "
          f"({100*b_is_final/n:.1f}%)")
    print(f"  A is the final occurrence in {a_is_final}/{n} tunes "
          f"({100*a_is_final/n:.1f}%)")

    # A<->B alternation share: what fraction of tunes' occurrence sequences
    # (ignoring i/intro at the very front) are entirely built from {A,B}?
    ab_only = 0
    for d in data:
        labs = set(lab for lab, _, _ in d["occ"])
        if labs <= {"A", "B", "i"}:
            ab_only += 1
    print(f"  Tunes using only labels subset of {{i, A, B}}: {ab_only}/{n} "
          f"({100*ab_only/n:.1f}%)")

    # AABA-shaped: A B A (contiguous, in that occurrence order, ignoring
    # leading i and other trailing material) as a literal substring check
    aba_substr = sum(1 for s in occ_strings if "ABA" in s)
    print(f"  Occurrence string contains literal 'ABA' substring: "
          f"{aba_substr}/{n} ({100*aba_substr/n:.1f}%)")
    abab_substr = sum(1 for s in occ_strings if "ABAB" in s)
    print(f"  Occurrence string contains literal 'ABAB' substring: "
          f"{abab_substr}/{n} ({100*abab_substr/n:.1f}%)")


# ─────────────────────────────────────────────────────────────────────────
# Spot checks (for surprising numbers)
# ─────────────────────────────────────────────────────────────────────────


def spot_check(data, titles: list[str]) -> None:
    print("\n" + "=" * 78)
    print("SPOT CHECKS")
    print("=" * 78)
    by_title = {d["title"]: d for d in data}
    for title in titles:
        d = by_title.get(title)
        if d is None:
            print(f"  {title}: not found")
            continue
        occ_str = "".join(lab for lab, _, _ in d["occ"])
        print(f"\n  {title}  (ts={d['time_signature']}, {d['n_bars']} bars)")
        print(f"    occurrence string: {occ_str}")
        print(f"    occurrences (label, start_bar, length): {d['occ']}")


if __name__ == "__main__":
    coverage_check()
    data = load_corpus()
    corpus_overview(data)
    occurrence_lengths(data)
    within_tune_consistency(data)
    tail_behavior(data)
    sequence_language(data)
