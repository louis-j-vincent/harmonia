"""``add_song(chart_ref, youtube_ref)`` — one call to grow the dataset.

A song = an iReal chart (chord symbols) + a YouTube recording (audio). This
module wires:

  * **audio** — a clean, standalone yt-dlp fetcher (:class:`YouTubeFetcher`)
    written IN this package. It downloads best-quality audio to
    ``data/dataset_audio/`` (gitignored), keyed by video id, and REUSES an
    already-present local file (e.g. under ``docs/audio/``) instead of
    re-downloading. It does NOT touch ``harmonia_server.py`` or ``serving/*``.
  * **chart** — resolved via ``harmonia.irealb_fetcher`` (imported READ-ONLY;
    another lane is editing it). A local chart already in ``data/ireal/`` is
    passed straight through to the aligner; a free-text query is resolved to a
    community chart's metadata.

Then optionally hands the (chart, audio) pair to :func:`harvest_song`.

INTEGRATION NOTE. The Brick-0 aligner consumes a chart as ``(ireal_file,
tune_title)`` reading ``data/ireal/<ireal_file>.txt``. A chart fetched fresh
from ``irealb_fetcher`` (an ``irealb://`` URL) must first be persisted into that
on-disk format before the aligner can align it — that bridge is the one
remaining wiring step and is intentionally left to the reconcile point (the
irealb_fetcher lane is mid-edit). Songs whose chart is already in
``data/ireal/`` harvest end-to-end today; that is the path the demo and
add-on-demand use.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from .gate import GateConfig
from .harvest import REPO, harvest_song, HarvestResult

log = logging.getLogger("harmonia.dataset.ingest")

DATASET_AUDIO = REPO / "data" / "dataset_audio"      # gitignored download cache
REUSE_DIRS = [REPO / "docs" / "audio", DATASET_AUDIO]  # look here before downloading
IREAL_DIR = REPO / "data" / "ireal"

_YT_ID_RE = re.compile(r"(?:v=|/shorts/|youtu\.be/|/embed/)([A-Za-z0-9_-]{11})")


def slugify(text: str) -> str:
    """Filesystem-safe slug (matches docs/audio naming: lowercase, _-separated)."""
    s = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return s or "untitled"


def youtube_id(url_or_id: str) -> Optional[str]:
    """Extract the 11-char video id from a URL, or return it if already an id."""
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", url_or_id):
        return url_or_id
    m = _YT_ID_RE.search(url_or_id)
    return m.group(1) if m else None


# ═════════════════════════════════════════════════════════════════════════════
# Audio: standalone yt-dlp fetcher (no server / serving dependency)
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class YouTubeFetcher:
    """Minimal yt-dlp audio fetcher with a local cache + reuse.

    ``out_dir`` is where new downloads land (gitignored). ``reuse_dirs`` are
    scanned first — if a file for this video id (or a provided ``prefer_stem``)
    already exists, it is reused and nothing is downloaded.
    """
    out_dir: Path = DATASET_AUDIO
    reuse_dirs: tuple[Path, ...] = tuple(REUSE_DIRS)
    audio_format: str = "bestaudio[ext=m4a]/bestaudio/best"

    def _find_cached(self, vid: Optional[str], prefer_stem: Optional[str]) -> Optional[Path]:
        stems = [s for s in (vid, prefer_stem) if s]
        for d in self.reuse_dirs:
            if not d.exists():
                continue
            for f in d.iterdir():
                if not f.is_file() or f.suffix.lower() not in (".m4a", ".mp3", ".wav", ".opus", ".webm"):
                    continue
                if f.stem in stems or (vid and vid in f.stem):
                    return f
        return None

    def fetch(self, youtube_ref: str, *, prefer_stem: Optional[str] = None,
              download: bool = True) -> Path:
        """Return a local audio path for ``youtube_ref`` (URL or 11-char id).

        Reuses an existing local file if present; otherwise downloads with
        yt-dlp. Raises ``RuntimeError`` if download is needed but disabled or
        yt-dlp/network is unavailable.
        """
        vid = youtube_id(youtube_ref)
        cached = self._find_cached(vid, prefer_stem)
        if cached is not None:
            log.info("reuse cached audio %s", cached)
            return cached
        if not download:
            raise RuntimeError(f"no cached audio for {youtube_ref!r} and download=False")

        try:
            import yt_dlp  # noqa: F401
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("yt-dlp not installed") from e

        self.out_dir.mkdir(parents=True, exist_ok=True)
        opts = {
            "format": self.audio_format,
            "outtmpl": str(self.out_dir / "%(id)s.%(ext)s"),
            "quiet": True, "no_warnings": True, "noplaylist": True,
            "postprocessors": [],
        }
        import yt_dlp
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(youtube_ref, download=True)
            path = Path(ydl.prepare_filename(info))
        if not path.exists():
            # fall back to whatever landed with this id
            hits = list(self.out_dir.glob(f"{info.get('id','')}.*"))
            if not hits:
                raise RuntimeError(f"yt-dlp reported success but no file for {youtube_ref!r}")
            path = hits[0]
        log.info("downloaded %s -> %s", youtube_ref, path)
        return path


# ═════════════════════════════════════════════════════════════════════════════
# Chart resolution (irealb_fetcher, read-only)
# ═════════════════════════════════════════════════════════════════════════════

ChartRef = Union[dict, str]


def resolve_chart(chart_ref: ChartRef) -> dict:
    """Resolve a chart reference into ``{ireal_file, tune_title, ...}``.

    * ``dict`` with ``ireal_file`` + ``tune_title`` — a chart already on disk in
      ``data/ireal/``; passed straight through to the aligner.
    * ``str`` — a free-text query resolved via
      ``harmonia.irealb_fetcher.search_community`` (community charts). Returns
      the best match's metadata + ``irealb_url``; see the module INTEGRATION
      NOTE on persisting it for alignment.
    """
    if isinstance(chart_ref, dict):
        if "ireal_file" in chart_ref and "tune_title" in chart_ref:
            local = IREAL_DIR / f"{chart_ref['ireal_file']}.txt"
            return {**chart_ref, "kind": "local",
                    "available": local.exists(), "path": str(local)}
        raise ValueError("dict chart_ref needs 'ireal_file' and 'tune_title'")
    # free-text query -> community search
    from harmonia import irealb_fetcher  # READ-ONLY import
    hits = irealb_fetcher.search_community(chart_ref, max_results=5)
    if not hits:
        raise LookupError(f"no community chart found for {chart_ref!r}")
    best = hits[0]
    return {"kind": "community", "tune_title": best.get("title"),
            "irealb_url": best.get("irealb_url"), "meta": best,
            "available": False,
            "note": "fetched chart must be persisted to data/ireal/ before the "
                    "aligner can consume it (see module INTEGRATION NOTE)"}


# ═════════════════════════════════════════════════════════════════════════════
# One-call entry
# ═════════════════════════════════════════════════════════════════════════════

def add_song(chart_ref: ChartRef, youtube_ref: str, *,
             song_id: Optional[str] = None, title: Optional[str] = None,
             do_harvest: bool = True, download: bool = True,
             gate_cfg: GateConfig = GateConfig()
             ) -> dict:
    """Add one song to the dataset: fetch audio + resolve chart, then (default)
    harvest it through the gate.

    Returns ``{song, chart, audio_path, harvest}`` where ``harvest`` is a
    :class:`HarvestResult` (or ``None`` if ``do_harvest=False`` or the chart is
    not yet locally alignable).
    """
    chart = resolve_chart(chart_ref)
    tune_title = title or chart.get("tune_title") or "Untitled"
    sid = song_id or slugify(tune_title)

    fetcher = YouTubeFetcher()
    audio = fetcher.fetch(youtube_ref, prefer_stem=sid, download=download)

    song = dict(song_id=sid, title=tune_title, audio=str(audio),
                ireal_file=chart.get("ireal_file"), tune_title=chart.get("tune_title"))
    # carry optional timing hints if the caller supplied them via chart_ref dict
    for k in ("human_anchor", "onset_nudge", "scored_end"):
        if isinstance(chart_ref, dict) and k in chart_ref:
            song[k] = chart_ref[k]

    result: Optional[HarvestResult] = None
    if do_harvest and chart.get("kind") == "local" and chart.get("available"):
        result = harvest_song(song, gate_cfg=gate_cfg)
    elif do_harvest:
        log.warning("chart for %s not locally alignable yet (%s) — audio fetched, "
                    "harvest skipped", sid, chart.get("kind"))

    return dict(song=song, chart=chart, audio_path=str(audio), harvest=result)


__all__ = ["YouTubeFetcher", "resolve_chart", "add_song", "slugify", "youtube_id",
           "DATASET_AUDIO", "IREAL_DIR"]
