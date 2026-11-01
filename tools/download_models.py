"""BUILD-TIME tool: downloads the local AI models once, so they can be shipped inside the app.

The app itself never downloads anything (and blocks all network access). Run this on a build machine:

    pip install huggingface_hub
    python tools/download_models.py --out models                 # recommended set
    python tools/download_models.py --out models --text large    # bigger, slower, a bit more accurate
    python tools/download_models.py --out models --no-llm        # skip the folder-naming model

Result:
    models/text/   multilingual sentence model (ONNX)       ~280 MB (base, int8)
    models/clip/   image understanding model (ONNX)         ~600 MB
    models/llm/    small instruction model (GGUF)           ~1.1 GB  (optional)
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TEXT = {
    # (repository, onnx file, query prefix, max tokens) - tried in order until one works
    "base": [("Xenova/multilingual-e5-base", "onnx/model_quantized.onnx", "query: ", 512),
             ("Xenova/multilingual-e5-small", "onnx/model_quantized.onnx", "query: ", 512),
             ("Xenova/paraphrase-multilingual-MiniLM-L12-v2", "onnx/model_quantized.onnx", "", 128)],
    "small": [("Xenova/multilingual-e5-small", "onnx/model_quantized.onnx", "query: ", 512),
              ("Xenova/paraphrase-multilingual-MiniLM-L12-v2", "onnx/model_quantized.onnx", "", 128)],
    "large": [("Xenova/multilingual-e5-large", "onnx/model_quantized.onnx", "query: ", 512),
              ("Xenova/multilingual-e5-base", "onnx/model_quantized.onnx", "query: ", 512)],
}
CLIP = [("Xenova/clip-vit-base-patch16", "onnx/vision_model.onnx", "onnx/text_model.onnx"),
        ("Xenova/clip-vit-base-patch32", "onnx/vision_model.onnx", "onnx/text_model.onnx")]
LLM = {
    "1.5b": [("Qwen/Qwen2.5-1.5B-Instruct-GGUF", "qwen2.5-1.5b-instruct-q4_k_m.gguf"),
             ("bartowski/Qwen2.5-1.5B-Instruct-GGUF", "Qwen2.5-1.5B-Instruct-Q4_K_M.gguf")],
    "3b": [("Qwen/Qwen2.5-3B-Instruct-GGUF", "qwen2.5-3b-instruct-q4_k_m.gguf"),
           ("bartowski/Qwen2.5-3B-Instruct-GGUF", "Qwen2.5-3B-Instruct-Q4_K_M.gguf")],
}


def _get(repo: str, filename: str, tmp: Path) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id=repo, filename=filename, local_dir=str(tmp)))


def fetch_text(out: Path, size: str) -> dict:
    from filesorter.models import OnnxTextEmbedder

    for repo, onnx_file, prefix, max_len in TEXT[size]:
        target = out / "text"
        shutil.rmtree(target, ignore_errors=True)
        target.mkdir(parents=True)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                shutil.copy(_get(repo, onnx_file, Path(tmp)), target / "model.onnx")
                shutil.copy(_get(repo, "tokenizer.json", Path(tmp)), target / "tokenizer.json")
            manifest = {"kind": "text", "name": repo.split("/")[-1], "source": repo, "onnx": "model.onnx",
                        "tokenizer": "tokenizer.json", "prefix": prefix, "pooling": "mean", "max_length": max_len}
            (target / "filesorter_model.json").write_text(json.dumps(manifest, indent=2))
            emb = OnnxTextEmbedder(target, threads=2)
            v = emb.embed(["Facture fournisseur et paiement", "Supplier invoice and payment",
                           "Photos de la soirée d'équipe au bowling"])
            same, other = float(v[0] @ v[1]), float(v[0] @ v[2])
            print(f"text model {repo}: FR/EN same meaning {same:.3f} vs unrelated {other:.3f}")
            if same <= other:
                raise RuntimeError("model sanity check failed")
            return manifest
        except Exception as exc:  # noqa: BLE001
            print(f"  could not use {repo}: {exc}")
    raise SystemExit("no text model could be downloaded")


def fetch_clip(out: Path) -> dict:
    from PIL import Image, ImageDraw

    from filesorter.models import ClipModel

    for repo, vision, text in CLIP:
        target = out / "clip"
        shutil.rmtree(target, ignore_errors=True)
        target.mkdir(parents=True)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                shutil.copy(_get(repo, vision, Path(tmp)), target / "vision_model.onnx")
                shutil.copy(_get(repo, text, Path(tmp)), target / "text_model.onnx")
                shutil.copy(_get(repo, "tokenizer.json", Path(tmp)), target / "tokenizer.json")
                pre = json.loads(_get(repo, "preprocessor_config.json", Path(tmp)).read_text())
            size = pre.get("crop_size", 224)
            size = size.get("height", 224) if isinstance(size, dict) else size
            manifest = {"kind": "clip", "name": repo.split("/")[-1], "source": repo,
                        "vision_onnx": "vision_model.onnx", "text_onnx": "text_model.onnx",
                        "tokenizer": "tokenizer.json", "image_size": int(size),
                        "mean": pre.get("image_mean"), "std": pre.get("image_std"), "context_length": 77}
            (target / "filesorter_model.json").write_text(json.dumps(manifest, indent=2))
            clip = ClipModel(target, threads=2)
            img = Image.new("RGB", (800, 600), "white")
            d = ImageDraw.Draw(img)
            for r in range(12):
                d.line([(0, 40 + r * 45), (800, 40 + r * 45)], fill="gray")
                for c in range(6):
                    d.text((10 + c * 130, 50 + r * 45), f"{(r + 1) * (c + 3) * 17}", fill="black")
            t = clip.embed_texts(["a screenshot of a spreadsheet", "a dog or a cat"])
            v = clip.embed_images([img])[0]
            print(f"image model {repo}: spreadsheet {float(t[0] @ v):.3f} vs pet {float(t[1] @ v):.3f}")
            if not float(t[0] @ v) > float(t[1] @ v):
                raise RuntimeError("model sanity check failed")
            return manifest
        except Exception as exc:  # noqa: BLE001
            print(f"  could not use {repo}: {exc}")
    raise SystemExit("no image model could be downloaded")


def fetch_llm(out: Path, size: str) -> str | None:
    for repo, fname in LLM[size]:
        target = out / "llm"
        target.mkdir(parents=True, exist_ok=True)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                shutil.copy(_get(repo, fname, Path(tmp)), target / fname)
            print(f"naming model {repo}/{fname}")
            return fname
        except Exception as exc:  # noqa: BLE001
            print(f"  could not use {repo}: {exc}")
    print("WARNING: no naming model downloaded (folders will be named from keywords)")
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "models"))
    ap.add_argument("--text", choices=sorted(TEXT), default="base")
    ap.add_argument("--llm", choices=sorted(LLM), default="1.5b")
    ap.add_argument("--no-llm", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    info = {"text": fetch_text(out, a.text), "clip": fetch_clip(out)}
    if not a.no_llm:
        info["llm"] = fetch_llm(out, a.llm)
    (out / "MODELS.json").write_text(json.dumps(info, indent=2))
    print("done:", out)


if __name__ == "__main__":
    main()
