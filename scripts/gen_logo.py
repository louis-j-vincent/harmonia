"""Generate the Harmonia logo: lowercase italic "h" (Georgia Italic — the
same family used for tempo/expression markings like "lento", "allegro" in
engraved sheet music), black on the app's cream paper colour.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent.parent / "docs" / "logo"
PAPER = (247, 243, 233)   # --paper
INK = (28, 28, 28)        # --ink

FONT_PATH = "/System/Library/Fonts/Supplemental/Georgia Italic.ttf"


def make_logo(size: int, corner_frac: float = 0.0, letter: str = "h") -> Image.Image:
    img = Image.new("RGB", (size, size), PAPER)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(FONT_PATH, int(size * 0.72))
    bbox = draw.textbbox((0, 0), letter, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (size - w) / 2 - bbox[0]
    y = (size - h) / 2 - bbox[1]
    draw.text((x, y), letter, font=font, fill=INK)
    if corner_frac:
        mask = Image.new("L", (size, size), 0)
        mdraw = ImageDraw.Draw(mask)
        mdraw.rounded_rectangle([0, 0, size - 1, size - 1],
                                 radius=int(size * corner_frac), fill=255)
        rounded = Image.new("RGB", (size, size), (0, 0, 0))
        rounded.paste(img, (0, 0), mask)
        out = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        out.paste(img, (0, 0), mask)
        return out
    return img


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    make_logo(1024).save(OUT / "logo_square.png")
    make_logo(1024, corner_frac=0.22).save(OUT / "logo_rounded.png")
    print(f"wrote {OUT / 'logo_square.png'}")
    print(f"wrote {OUT / 'logo_rounded.png'}")


if __name__ == "__main__":
    main()
