"""iReal-Pro-style chord-chart rendering.

Takes a structured chart — a title/composer/key header plus a per-bar chord
timeline — and draws a clean lead-sheet grid: measures four-to-a-row, thin
barlines, boxed section letters, double barlines at section boundaries, and
properly typeset jazz chord symbols (big root, smaller quality, △ ø ° glyphs
and stacked ♭/♯ alterations).

The renderer is source-agnostic: feed it an accompaniment-db record
(``from_db_record``) or any list of ``BarChord`` entries, so the same look
serves both the ground-truth iReal charts and the model's inferred charts.

Design mirror of iReal Pro's default "jazz" theme:
    - warm cream paper, upright-italic chord font (matplotlib mathtext)
    - root letter large; quality (m, 7, sus…) smaller on the same baseline
    - accidentals and extensions as raised/stacked superscripts
    - section markers as a small outlined letter box at the bar's top-left
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# ── palette (iReal "jazz" theme, toned down) ────────────────────────────────────
PAPER = "#f7f3e9"      # warm cream
INK = "#1c1c1c"        # near-black chord ink
RULE = "#b9b09a"       # barlines / grid
FAINT = "#8a8371"      # subheader text
ACCENT = "#8a2b2b"     # section letters / final bar (muted red)

_NOTE_RE = re.compile(r"^([A-G])([b#]?)(.*)$")
# split a quality tail into (leading symbol run, alteration list) for stacking
_ALT_RE = re.compile(r"([b#])(\d+)")


@dataclass
class BarChord:
    """One chord placed in the grid.

    bar/beat are 0-indexed; ``beat`` positions the chord horizontally inside its
    bar (0 = downbeat). ``sub`` optionally colours the symbol (e.g. a low-
    confidence inferred chord) — None keeps the default ink.
    """

    bar: int
    beat: float
    symbol: str          # raw iReal token, e.g. "F-7", "Ab^7", "C7b9/G"
    colour: str | None = None


@dataclass
class Chart:
    title: str
    composer: str = ""
    key: str = ""
    style: str = ""
    tempo: int | None = None
    time_signature: tuple[int, int] = (4, 4)
    n_bars: int = 0
    section_per_bar: list[str] = field(default_factory=list)
    chords: list[BarChord] = field(default_factory=list)

    @classmethod
    def from_db_record(cls, rec: dict, use_ireal: bool = True) -> "Chart":
        """Build a Chart from an accomp-db record (see build_accompaniment_db)."""
        ts = rec.get("time_signature", [4, 4])
        chords = [
            BarChord(bar=e["bar"] - 1, beat=e.get("beat", 0),
                     symbol=e["ireal"] if use_ireal else e["mma"])
            for e in rec["chord_timeline"]
        ]
        return cls(
            title=rec.get("title", ""),
            composer=rec.get("composer", ""),
            key=rec.get("key", ""),
            style=rec.get("style", ""),
            tempo=rec.get("tempo"),
            time_signature=(ts[0], ts[1]),
            n_bars=rec.get("n_bars", 0),
            section_per_bar=rec.get("section_per_bar", []),
            chords=chords,
        )


# ── chord-symbol typesetting → matplotlib mathtext ──────────────────────────────
# The root letter is drawn separately (large); these return the *quality* body as
# a mathtext fragment (no leading root), split into a baseline part and a raised
# superscript part so alterations float while m/sus/dim sit on the line.
def _accidental(a: str) -> str:
    return {"b": r"\flat ", "#": r"\sharp "}.get(a, "")


def _typeset_quality(q: str) -> tuple[str, str]:
    """Return (baseline_mathtext, super_mathtext) for a raw iReal quality tail.

    Examples: ""→("",""); "-7"→("m","7"); "^7"→("",r"\triangle 7");
    "7b9"→("","7\\flat 9"); "h7"→(r"\varnothing","7"); "sus"→("sus4","").
    """
    if q == "":
        return "", ""

    # explicit special-cases where the glyph belongs on the baseline
    base = ""
    rest = q
    if rest.startswith("-^"):          # minor-major: m△…
        base, rest = "m", r"\triangle " + rest[2:]
    elif rest.startswith("-7b5"):      # half-diminished
        base, rest = r"\varnothing ", rest[4:]
    elif rest.startswith("-"):
        base, rest = "m", rest[1:]
    elif rest.startswith("h"):
        base, rest = r"\varnothing ", rest[1:]
    elif rest.startswith("o"):
        base, rest = r"\circ ", rest[1:]
    elif rest.startswith("^"):
        base, rest = "", r"\triangle " + rest[1:]
    elif rest.startswith("+"):
        base, rest = "+", rest[1:]
    elif rest.startswith("sus"):
        base, rest = "sus" + (rest[3:] or "4"), ""
    elif rest.startswith("5"):
        base, rest = "5", rest[1:]

    # anything containing sus later (e.g. 7sus) → keep sus on baseline, digits super
    if "sus" in rest:
        pre, _, post = rest.partition("sus")
        return base + ("sus" + (post or "4")), _alterations(pre)

    return base, _alterations(rest)


def _alterations(s: str) -> str:
    """Render a superscript run: digits verbatim, bN/#N → ♭N/♯N, 'alt' upright."""
    if not s:
        return ""
    s = s.replace("69", "6/9")
    s = _ALT_RE.sub(lambda m: _accidental(m.group(1)) + m.group(2), s)
    s = s.replace("alt", r"\mathrm{alt}").replace("add", r"\mathrm{add}")
    s = s.replace("^", r"\triangle ")
    return s


_MATH_TO_UNICODE = {
    r"\flat ": "♭", r"\sharp ": "♯", r"\triangle ": "△", r"\varnothing ": "ø",
    r"\circ ": "°", r"\mathrm{alt}": "alt", r"\mathrm{add}": "add",
}


def _demath(s: str) -> str:
    for k, v in _MATH_TO_UNICODE.items():
        s = s.replace(k, v)
    return s


def _wrap_acc(s: str) -> str:
    """Shrink ♭/♯ so they sit tight against the note letter, as in engraving."""
    return s.replace("♭", "<span class='acc'>♭</span>").replace("♯", "<span class='acc'>♯</span>")


def chord_html(symbol: str) -> str:
    """Typeset an iReal token as HTML: big root, smaller quality, raised
    extensions — the DOM twin of the matplotlib chord (for interactive charts)."""
    root, base, sup = _chord_mathtext(symbol)
    html = f'<span class="root">{_wrap_acc(_demath(root))}</span>'
    if base or sup:
        html += '<span class="qual">' + _wrap_acc(_demath(base))
        if sup:
            html += f"<sup>{_demath(sup)}</sup>"
        html += "</span>"
    return html


def _chord_mathtext(symbol: str) -> tuple[str, str, str]:
    """Split a full iReal token into (root, quality_baseline, quality_super)
    mathtext fragments. Bass note (``/G``) is folded into the baseline part."""
    token, _, bass = symbol.partition("/")
    m = _NOTE_RE.match(token.strip())
    if not m:
        return token, "", ""
    letter, acc, qual = m.groups()
    root = letter + _accidental(acc)
    base, sup = _typeset_quality(qual)
    if bass:
        bm = _NOTE_RE.match(bass)
        bass_str = (bm.group(1) + _accidental(bm.group(2))) if bm else bass
        base = base + r"/" + bass_str
    return root, base, sup


def _draw_chord(ax, x, y, symbol, size, colour):
    """Draw a chord centred at (x, y) in axes coords; root large, quality smaller
    on the same baseline, extensions raised. Returns nothing."""
    root, base, sup = _chord_mathtext(symbol)
    fig = ax.figure
    rend = fig.canvas.get_renderer()
    inv = ax.transAxes.inverted()

    def width(s, fs):
        if not s:
            return 0.0
        t = ax.text(0, 0, f"${s}$", fontsize=fs, transform=ax.transAxes,
                    ha="left", va="baseline")
        fig.canvas.draw()
        bb = t.get_window_extent(rend)
        w = inv.transform((bb.width, 0))[0] - inv.transform((0, 0))[0]
        t.remove()
        return w

    qual = base + (("^{" + sup + "}") if sup else "")
    w_root = width(root, size)
    w_qual = width(qual, size * 0.62)
    total = w_root + w_qual
    x0 = x - total / 2
    ax.text(x0, y, f"${root}$", fontsize=size, color=colour, ha="left",
            va="baseline", transform=ax.transAxes)
    if qual:
        ax.text(x0 + w_root, y, f"${qual}$", fontsize=size * 0.62, color=colour,
                ha="left", va="baseline", transform=ax.transAxes)


def render_chart(chart: Chart, out_path: str | Path, bars_per_row: int = 4,
                 dpi: int = 200, caption: str = "") -> Path:
    """Render ``chart`` to a PNG at ``out_path`` and return the path.

    ``caption`` is an optional faint footer line (e.g. a legend for chord tints).
    """
    out_path = Path(out_path)
    n_bars = max(chart.n_bars, (max((c.bar for c in chart.chords), default=-1) + 1))
    n_rows = max(1, math.ceil(n_bars / bars_per_row))

    # geometry: fix row height and header band in *inches*, derive the rest so a
    # 4- or 16-row chart keeps the same compact, wide iReal proportions.
    fig_w = 9.0
    row_in = 0.82
    header_in = 1.25
    pad_in = 0.35
    fig_h = header_in + n_rows * row_in + pad_in

    header_h = header_in / fig_h
    pad_h = pad_in / fig_h
    grid_top = 1 - header_h
    row_h = (grid_top - pad_h) / n_rows
    left, right = 0.045, 0.955
    cell_w = (right - left) / bars_per_row

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor=PAPER)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_facecolor(PAPER)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # ── header (positioned in inches from the top so it never crowds the grid) ──
    def y_from_top(inch):
        return 1 - inch / fig_h

    ax.text(0.5, y_from_top(0.35), chart.title, fontsize=21, weight="bold",
            color=INK, ha="center", va="top", family="serif")
    sub = []
    if chart.key:
        sub.append(f"Key {chart.key}")
    if chart.style:
        sub.append(chart.style)
    if chart.tempo:
        sub.append(f"♩ = {chart.tempo}")
    if sub:
        ax.text(left, y_from_top(0.85), "   ".join(sub), fontsize=10.5,
                color=FAINT, ha="left", va="top", style="italic")
    if chart.composer:
        ax.text(right, y_from_top(0.85), chart.composer, fontsize=10.5,
                color=FAINT, ha="right", va="top", style="italic")

    # chords indexed by bar
    by_bar: dict[int, list[BarChord]] = {}
    for c in chart.chords:
        by_bar.setdefault(c.bar, []).append(c)

    bpb = chart.time_signature[0] or 4
    spb = chart.section_per_bar

    def section_of(bar: int) -> str:
        return spb[bar] if 0 <= bar < len(spb) else ""

    chord_size = 21 if bars_per_row <= 4 else 17
    gap = 0.14 * row_h            # separates each row-strip, iReal style

    for bar in range(n_bars):
        row = bar // bars_per_row
        col = bar % bars_per_row
        x0 = left + col * cell_w
        x1 = x0 + cell_w
        y_top = grid_top - row * row_h - gap
        y_bot = y_top - (row_h - 2 * gap)
        y_mid = (y_top + y_bot) / 2

        new_section = bar == 0 or section_of(bar) != section_of(bar - 1)
        # left barline of the cell (double if a section starts here)
        _barline(ax, x0, y_bot, y_top, heavy=new_section and col != 0,
                 double=new_section)

        # section letter box at the bar's top-left
        if new_section and section_of(bar):
            _section_box(ax, x0, y_top, section_of(bar))

        # chords in this bar, positioned by beat
        cs = sorted(by_bar.get(bar, []), key=lambda c: c.beat)
        for i, c in enumerate(cs):
            if len(cs) == 1:
                cx = (x0 + x1) / 2
            else:
                frac = (c.beat / bpb) if bpb else i / len(cs)
                cx = x0 + cell_w * (0.24 + 0.60 * frac)
            _draw_chord(ax, cx, y_mid - 0.012, c.symbol, chord_size,
                        c.colour or INK)

    # right-hand barline of every row (final bar of piece gets a heavy cap)
    for row in range(n_rows):
        last_col = min(bars_per_row, n_bars - row * bars_per_row)
        x = left + last_col * cell_w
        y_top = grid_top - row * row_h - gap
        y_bot = y_top - (row_h - 2 * gap)
        is_final = row == n_rows - 1
        _barline(ax, x, y_bot, y_top, heavy=is_final, double=is_final,
                 colour=ACCENT if is_final else RULE)

    if caption:
        ax.text(left, 0.4 * pad_h, caption, fontsize=8.5, color=FAINT,
                ha="left", va="center", style="italic", transform=ax.transAxes)

    fig.savefig(out_path, dpi=dpi, facecolor=PAPER)
    plt.close(fig)
    return out_path


def _barline(ax, x, y0, y1, heavy=False, double=False, colour=RULE):
    lw = 2.6 if heavy else 1.1
    ax.plot([x, x], [y0, y1], color=colour, lw=lw, solid_capstyle="butt",
            transform=ax.transAxes, zorder=1)
    if double:
        dx = 0.006
        ax.plot([x - dx, x - dx], [y0, y1], color=colour, lw=1.1,
                solid_capstyle="butt", transform=ax.transAxes, zorder=1)


def _section_box(ax, x, y_top, letter):
    s = 0.032
    box = FancyBboxPatch((x + 0.004, y_top - s - 0.004), s, s,
                         boxstyle="round,pad=0.001,rounding_size=0.004",
                         linewidth=1.2, edgecolor=ACCENT, facecolor="none",
                         transform=ax.transAxes, zorder=3)
    ax.add_patch(box)
    ax.text(x + 0.004 + s / 2, y_top - 0.004 - s / 2, letter, fontsize=10,
            weight="bold", color=ACCENT, ha="center", va="center",
            transform=ax.transAxes, zorder=4)
