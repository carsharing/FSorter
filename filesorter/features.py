"""Content fingerprints used to find duplicates: file hashes, perceptual image hashes, text fingerprints."""

from __future__ import annotations

import hashlib
import os
import re
import unicodedata

import numpy as np

from .paths import long_path

_CHUNK = 1 << 20


def head_tail_hash(path, size: int) -> str:
    """Cheap pre-filter: size + first and last 64 KB."""
    h = hashlib.blake2b(digest_size=16)
    h.update(str(size).encode())
    with open(long_path(path), "rb") as f:
        h.update(f.read(65536))
        if size > 131072:
            f.seek(-65536, os.SEEK_END)
            h.update(f.read(65536))
    return h.hexdigest()


def full_hash(path) -> str:
    h = hashlib.blake2b(digest_size=20)
    with open(long_path(path), "rb") as f:
        while chunk := f.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def text_fingerprint(text: str, min_chars: int = 300) -> str | None:
    """Same words in the same order => same fingerprint, whatever the file name, format or line breaks."""
    t = unicodedata.normalize("NFKC", text or "").lower()
    t = re.sub(r"[\W_]+", "", t)
    if len(t) < min_chars:
        return None
    return hashlib.blake2b(t[:200_000].encode("utf-8"), digest_size=16).hexdigest()


# ---------------------------------------------------------------------------------------------------------------
# Perceptual hashes (robust to resizing, recompression, small colour changes)
# ---------------------------------------------------------------------------------------------------------------
def _dct_matrix(n: int) -> np.ndarray:
    k = np.arange(n)[:, None]
    i = np.arange(n)[None, :]
    m = np.cos(np.pi * (2 * i + 1) * k / (2 * n)) * np.sqrt(2 / n)
    m[0] /= np.sqrt(2)
    return m


_DCT32 = _dct_matrix(32)


def _bits_to_int(bits: np.ndarray) -> int:
    v = 0
    for b in bits.ravel():
        v = (v << 1) | int(bool(b))
    # store as signed 64-bit so it fits in SQLite INTEGER
    return v - (1 << 64) if v >= (1 << 63) else v


def colour_signature(img) -> bytes:
    """Average colour of a 4x4 grid (48 bytes). Tells apart pictures with the same shapes but other colours."""
    from PIL import Image

    return np.asarray(img.convert("RGB").resize((4, 4), Image.Resampling.BOX), dtype=np.uint8).tobytes()


def image_hashes(img) -> tuple[int, int, float]:
    """(pHash, dHash, detail) for a PIL image. `detail` is low for blank/uniform images (unreliable hashes)."""
    from PIL import Image

    gray = img.convert("L")
    a32 = np.asarray(gray.resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float64)
    detail = float(a32.std())
    dct = _DCT32 @ a32 @ _DCT32.T
    low = dct[:8, :8].copy()
    med = np.median(low.ravel()[1:])
    phash = _bits_to_int(low > med)
    a9 = np.asarray(gray.resize((9, 8), Image.Resampling.LANCZOS), dtype=np.int16)
    dhash = _bits_to_int(a9[:, 1:] > a9[:, :-1])
    return phash, dhash, detail


def to_u64(values) -> np.ndarray:
    return np.asarray(values, dtype=np.int64).view(np.uint64)


if hasattr(np, "bitwise_count"):
    def popcount(x: np.ndarray) -> np.ndarray:
        return np.bitwise_count(x)
else:  # pragma: no cover - numpy < 2
    _POP8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

    def popcount(x: np.ndarray) -> np.ndarray:
        b = x.view(np.uint8).reshape(x.shape + (8,))
        return _POP8[b].sum(-1)


def hamming_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return popcount(a[:, None] ^ b[None, :])
