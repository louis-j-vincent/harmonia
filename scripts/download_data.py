"""
Download and prepare training datasets.

Datasets:
  POP909   — 909 pop songs with chord annotations. ~500MB. Free, no auth needed.
  MAESTRO  — 200h aligned piano audio+MIDI. ~120GB. Requires accepting licence.
  GiantMIDI — 10k piano MIDIs. ~2GB. Free.

Usage:
    python scripts/download_data.py pop909
    python scripts/download_data.py maestro   # warns about size
    python scripts/download_data.py giantmidi
    python scripts/download_data.py all
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    print(f"  $ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd, check=True)


def _download(url: str, dest: Path) -> None:
    print(f"  Downloading {url.split('/')[-1]}...")
    urllib.request.urlretrieve(url, dest)
    print(f"  Saved → {dest}")


# ── POP909 ─────────────────────────────────────────────────────────────────

def download_pop909() -> None:
    dest = DATA_DIR / "pop909"
    if dest.exists() and any(dest.iterdir()):
        print(f"POP909 already at {dest}. Delete it to re-download.")
        return
    print("Cloning POP909 dataset (~500MB)...")
    _run(["git", "clone", "--depth=1",
          "https://github.com/music-x-lab/POP909-Dataset",
          str(dest)])
    print(f"✓ POP909 ready at {dest}")
    print()
    print("Next: render MIDIs to audio using FluidSynth:")
    print("  python scripts/render_pop909.py")


# ── GiantMIDI ──────────────────────────────────────────────────────────────

def download_giantmidi() -> None:
    dest = DATA_DIR / "giantmidi"
    if dest.exists() and any(dest.iterdir()):
        print(f"GiantMIDI already at {dest}.")
        return
    dest.mkdir(parents=True, exist_ok=True)
    print("Cloning GiantMIDI-Piano (~2GB)...")
    _run(["git", "clone", "--depth=1",
          "https://github.com/bytedance/GiantMIDI-Piano",
          str(dest)])
    print(f"✓ GiantMIDI ready at {dest}")


# ── MAESTRO ────────────────────────────────────────────────────────────────

def download_maestro() -> None:
    print("MAESTRO v3.0.0 (~120GB uncompressed, ~16GB compressed)")
    print("Requires accepting the Creative Commons licence at:")
    print("  https://magenta.tensorflow.org/datasets/maestro")
    print()
    answer = input("Have you accepted the licence? [y/N] ").strip().lower()
    if answer != "y":
        print("Aborted. Accept the licence first.")
        return

    dest = DATA_DIR / "maestro"
    dest.mkdir(parents=True, exist_ok=True)
    url = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.zip"
    zip_path = DATA_DIR / "maestro-v3.0.0.zip"

    print(f"Downloading MAESTRO (~16GB) → {zip_path} ...")
    print("This will take a while. Go record some piano.")
    _download(url, zip_path)

    print("Extracting...")
    shutil.unpack_archive(str(zip_path), str(dest))
    zip_path.unlink()
    print(f"✓ MAESTRO ready at {dest}")


# ── Render POP909 MIDIs ────────────────────────────────────────────────────

def render_pop909(n_variants: int = 5, max_songs: int | None = None) -> None:
    """
    Render POP909 MIDIs to audio using FluidSynth.
    Requires FluidSynth + at least one soundfont in data/soundfonts/.
    """
    from harmonia.data.midi_renderer import MIDIRenderer
    from harmonia.data.pop909_parser import POP909Parser

    pop909_dir = DATA_DIR / "pop909"
    render_dir = DATA_DIR / "renders" / "pop909"
    soundfont_dir = DATA_DIR / "soundfonts"

    if not pop909_dir.exists():
        print("POP909 not found. Run: python scripts/download_data.py pop909")
        return

    if not list(soundfont_dir.glob("*.sf2")):
        print(f"No soundfonts found in {soundfont_dir}/")
        print("Download a free soundfont:")
        print("  Salamander Grand Piano: https://freepats.zenvoid.org/Piano/")
        print("  GeneralUser GS:         https://schristiancollins.com/generaluser.php")
        return

    parser = POP909Parser(pop909_dir)
    songs = parser.parse_all(max_songs=max_songs)
    renderer = MIDIRenderer(soundfont_dir)

    print(f"Rendering {len(songs)} songs × {n_variants} variants...")
    for i, song in enumerate(songs):
        out_dir = render_dir / song.song_id
        out_dir.mkdir(parents=True, exist_ok=True)
        variants = renderer.render_variants(
            midi_path=song.midi_path,
            output_dir=out_dir,
            n_variants=n_variants,
        )
        print(f"  [{i+1}/{len(songs)}] {song.song_id}: {len(variants)} renders")

    print(f"✓ Renders saved to {render_dir}")


# ── CLI ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Download Harmonia training data")
    parser.add_argument(
        "dataset",
        choices=["pop909", "maestro", "giantmidi", "all", "render-pop909"],
        help="Dataset to download (or 'render-pop909' to render MIDIs)"
    )
    parser.add_argument(
        "--variants", type=int, default=5,
        help="Number of timbral variants per MIDI (for render-pop909)"
    )
    parser.add_argument(
        "--max-songs", type=int, default=None,
        help="Limit number of songs (for testing)"
    )
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if args.dataset == "pop909" or args.dataset == "all":
        download_pop909()
    if args.dataset == "giantmidi" or args.dataset == "all":
        download_giantmidi()
    if args.dataset == "maestro" or args.dataset == "all":
        download_maestro()
    if args.dataset == "render-pop909":
        render_pop909(n_variants=args.variants, max_songs=args.max_songs)


if __name__ == "__main__":
    main()
