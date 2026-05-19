"""Render 10 single-digit test images for the e2e classifier check.

Downloads Indie Flower (Google Fonts, OFL) and renders digits 0-9 as
28x28 grayscale PNGs — the same size the trained split model consumes.
A 280x280 preview is also written for each digit so the rendering is
easy to inspect by eye.
"""

import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FONT_URL = (
    "https://raw.githubusercontent.com/google/fonts/main/"
    "ofl/indieflower/IndieFlower-Regular.ttf"
)


def main() -> None:
    data_path = Path(__file__).resolve().parent.parent / "data"
    font_dir = data_path / "fonts"
    img_dir = data_path / "test-images"
    font_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)

    font_path = font_dir / "IndieFlower-Regular.ttf"
    if not font_path.exists():
        print(f"Downloading {FONT_URL} -> {font_path}")
        urllib.request.urlretrieve(FONT_URL, font_path)

    # Render on a generous canvas, then crop to the actual ink bounding box
    # (the font's logical bbox is much taller than its glyph for Indie
    # Flower) before centring at MNIST's ~20px-in-28px target envelope.
    big = 400
    font = ImageFont.truetype(str(font_path), 280)

    for digit in range(10):
        scratch = Image.new("L", (big, big), color=0)
        ImageDraw.Draw(scratch).text((big / 2, big / 2), str(digit), fill=255, font=font, anchor="mm")
        ink_bbox = scratch.getbbox()  # (l, t, r, b) of non-zero pixels
        glyph = scratch.crop(ink_bbox)

        # Fit the glyph into a 20x20 inner box, preserving aspect ratio.
        gw, gh = glyph.size
        target = 20
        if gw >= gh:
            new_w, new_h = target, max(1, round(target * gh / gw))
        else:
            new_w, new_h = max(1, round(target * gw / gh)), target
        glyph_28 = glyph.resize((new_w, new_h), Image.LANCZOS)

        out = Image.new("L", (28, 28), color=0)
        out.paste(glyph_28, ((28 - new_w) // 2, (28 - new_h) // 2))

        out.save(img_dir / f"digit-{digit}.png")
        out.resize((280, 280), Image.NEAREST).save(
            img_dir / f"digit-{digit}-preview.png"
        )

    print(f"Wrote {2 * 10} files to {img_dir}/")


if __name__ == "__main__":
    main()
