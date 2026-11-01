"""The app icon, drawn in code (no image files to ship)."""

from __future__ import annotations

import io


def draw(size: int = 256):
    from PIL import Image, ImageDraw

    s = size
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([s * 0.04, s * 0.04, s * 0.96, s * 0.96], radius=s * 0.2, fill=(37, 99, 235, 255))
    # folder
    d.rounded_rectangle([s * 0.18, s * 0.30, s * 0.48, s * 0.42], radius=s * 0.04, fill=(255, 214, 102, 255))
    d.rounded_rectangle([s * 0.18, s * 0.36, s * 0.82, s * 0.78], radius=s * 0.06, fill=(255, 196, 61, 255))
    # three sorted "lines"
    for k, w in enumerate((0.44, 0.34, 0.24)):
        y = s * (0.47 + k * 0.1)
        d.rounded_rectangle([s * 0.28, y, s * (0.28 + w), y + s * 0.055], radius=s * 0.03, fill=(37, 99, 235, 255))
    return img


def png_bytes(size: int = 256) -> bytes:
    buf = io.BytesIO()
    draw(size).save(buf, format="PNG")
    return buf.getvalue()


def save_ico(path) -> None:
    draw(256).save(path, format="ICO", sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
