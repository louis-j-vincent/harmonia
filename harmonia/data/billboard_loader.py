"""Billboard McGill dataset loader — ground truth for training/evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

try:
    import mirdata
except ImportError:
    mirdata = None

from harmonia.data.billboard_translator import billboard_chord_list_to_harmonia


class BillboardDataset:
    """Wrapper around mirdata's Billboard dataset with Harmonia format conversion."""

    def __init__(self, chord_type: str = "majmin"):
        """Initialize Billboard dataset via mirdata.
        
        Args:
            chord_type: which chord annotation to load ("majmin", "full", etc.)
        
        Raises:
            ImportError: if mirdata is not installed
            FileNotFoundError: if Billboard index is not downloaded
        """
        if mirdata is None:
            raise ImportError("mirdata not installed; run: pip install mirdata")

        self.dataset = mirdata.initialize("billboard")
        self.chord_type = chord_type

        if not self.dataset.track_ids:
            raise FileNotFoundError(
                "Billboard dataset index not found. Run: "
                "billboard_dataset = mirdata.initialize('billboard'); "
                "billboard_dataset.download()"
            )

    def __len__(self) -> int:
        """Total number of tracks in Billboard."""
        return len(self.dataset.track_ids)

    def track_ids(self) -> list[str]:
        """List of all Billboard track IDs."""
        return self.dataset.track_ids

    def load_track_gt(self, track_id: str) -> dict:
        """Load a single track's ground truth in Harmonia format."""
        track = self.dataset.track(track_id)

        chord_attr = f"chords_{self.chord_type}"
        if not hasattr(track, chord_attr):
            raise ValueError(f"Track does not have {chord_attr}")

        chord_data = getattr(track, chord_attr)
        if chord_data is None:
            chords = []
        else:
            chords = billboard_chord_list_to_harmonia(chord_data)

        return {
            "song_id": track_id,
            "title": track.title or "",
            "artist": track.artist or "",
            "chords": chords,
            "audio_path": track.audio_path if hasattr(track, "audio_path") else None,
            "metadata": {
                "chart_date": getattr(track, "chart_date", None),
                "peak_rank": getattr(track, "peak_rank", None),
                "weeks_on_chart": getattr(track, "weeks_on_chart", None),
            },
        }

    def load_all_tracks_gt(self) -> list[dict]:
        """Load all Billboard tracks' ground truth."""
        result = []
        for track_id in self.dataset.track_ids:
            try:
                gt = self.load_track_gt(track_id)
                result.append(gt)
            except Exception as e:
                print(f"Warning: failed to load track {track_id}: {e}")
                continue
        return result

    def export_to_jsonl(self, output_path: Path) -> None:
        """Export all tracks to JSONL format (one track per line)."""
        output_path = Path(output_path)
        with open(output_path, "w") as f:
            for track_id in self.dataset.track_ids:
                try:
                    gt = self.load_track_gt(track_id)
                    valid_chords = [c for c in gt["chords"] if c["root"] is not None]
                    if valid_chords:
                        gt["chords"] = valid_chords
                        f.write(json.dumps(gt) + "\n")
                except Exception as e:
                    print(f"Warning: skipping track {track_id}: {e}")

    def split_train_val_test(
        self, train_ratio: float = 0.8, val_ratio: float = 0.1, seed: int = 42
    ) -> tuple[list[str], list[str], list[str]]:
        """Create deterministic train/val/test split of track IDs."""
        import random

        random.seed(seed)
        track_ids = list(self.dataset.track_ids)
        random.shuffle(track_ids)

        n = len(track_ids)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train = track_ids[:n_train]
        val = track_ids[n_train : n_train + n_val]
        test = track_ids[n_train + n_val :]

        return train, val, test


def load_billboard_gt_for_inference(
    track_id: str, chord_type: str = "majmin"
) -> list[dict]:
    """Convenience function: load a single track's GT for inference evaluation."""
    bb = BillboardDataset(chord_type=chord_type)
    track_gt = bb.load_track_gt(track_id)
    return track_gt["chords"]
