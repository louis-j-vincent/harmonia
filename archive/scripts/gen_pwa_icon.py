"""Generate PWA / apple-touch-icon PNGs for the Harmonia home-screen app —
the same italic-h logo as docs/logo/ (see scripts/gen_logo.py). No pre-
rounded corners: iOS applies its own mask to home-screen icons."""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent.parent / "docs" / "pwa"
PAPER = (247, 243, 233)   # --paper
INK = (28, 28, 28)        # --ink
FONT_PATH = "/System/Library/Fonts/Supplemental/Georgia Italic.ttf"


def make_icon(size: int) -> Image.Image:
    img = Image.new("RGB", (size, size), PAPER)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(FONT_PATH, int(size * 0.72))
    bbox = draw.textbbox((0, 0), "h", font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (size - w) / 2 - bbox[0]
    y = (size - h) / 2 - bbox[1]
    draw.text((x, y), "h", font=font, fill=INK)
    return img


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for size, name in [(180, "apple-touch-icon.png"), (192, "icon-192.png"), (512, "icon-512.png")]:
        make_icon(size).save(OUT / name)
        print(f"wrote {OUT / name}")


if __name__ == "__main__":
    main()
