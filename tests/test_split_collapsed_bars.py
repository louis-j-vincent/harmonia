"""Red-first tests for the _split_collapsed_bars_via_musx min_half bug.

Bug (found by the post-musx trace, docs/postmusx_segment_loss.md): the
splitter deletes every baseline chord inside a fast run, then re-emits
from music-x-lab SKIPPING any segment under min_half=1.4 beats — so a
correct 1-beat chord musx genuinely decoded is deleted and never
re-added (8 chords lost on Let It Be, all exactly 1.00 beat).

Fix under test: min_half exists to stop the run-boundary clipping from
emitting slivers, so it must apply only to segments the clipping
actually shortened; un-clipped segments get a 0.75-beat floor.
"""

from harmonia.models.chord_pipeline_v1 import _split_collapsed_bars_via_musx

PERIOD = 0.6  # 100 BPM


def _chord(label, t0, t1):
    return {
        "label": label,
        "start_s": t0,
        "end_s": t1,
        "confidence": 0.9,
        "suggestions": [],
        "duration_beats": max(1, round((t1 - t0) / PERIOD)),
    }


def test_unclipped_one_beat_musx_chord_survives():
    """A genuine 1.0-beat musx chord inside the run must be re-emitted."""
    mx = [
        (10.0, 11.2, "C:maj"),   # 2.0 beats — fast
        (11.2, 11.8, "D:min"),   # 1.0 beat — the victim, NOT clipped
        (11.8, 13.0, "C:maj"),
        (13.0, 14.2, "G:maj"),
        (14.2, 15.4, "F:maj"),
    ]
    chords = [_chord("C:maj", 10.0, 12.4), _chord("F:maj", 12.4, 15.4),
              _chord("A:min", 15.4, 17.8)]  # last one is outside the run
    _split_collapsed_bars_via_musx(chords, mx, PERIOD)
    labels = [(c["label"], c["start_s"]) for c in chords]
    assert ("D:min", 11.2) in labels, (
        "the un-clipped 1-beat D:min was deleted and never re-emitted: "
        f"{labels}"
    )
    assert ("A:min", 15.4) in labels  # outside the run: untouched


def test_clipped_sliver_still_dropped():
    """A fragment created by the run-boundary cut stays below the floor."""
    mx = [
        (8.0, 10.0, "E:min"),    # 3.3 beats — slow, straddles the run start
        (9.7, 10.9, "C:maj"),    # fast run begins
        (10.9, 12.1, "F:maj"),
        (12.1, 13.3, "G:maj"),
        (13.3, 14.5, "Bb:maj"),
    ]
    chords = [_chord("C:maj", 9.7, 12.1), _chord("G:maj", 12.1, 14.5)]
    _split_collapsed_bars_via_musx(chords, mx, PERIOD)
    # E:min clipped to (9.7, 10.0) = 0.5 beats: must NOT be emitted
    assert not any(c["label"] == "E:min" for c in chords), chords


def test_output_sorted_and_run_reemitted():
    mx = [
        (10.0, 11.2, "C:maj"),
        (11.2, 11.8, "D:min"),
        (11.8, 13.0, "C:maj"),
        (13.0, 14.2, "G:maj"),
        (14.2, 15.4, "F:maj"),
    ]
    chords = [_chord("C:maj", 10.0, 12.4), _chord("F:maj", 12.4, 15.4)]
    _split_collapsed_bars_via_musx(chords, mx, PERIOD)
    starts = [c["start_s"] for c in chords]
    assert starts == sorted(starts)
    assert len(chords) == 5
