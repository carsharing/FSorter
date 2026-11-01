"""Local AI models. Nothing here downloads anything: models are read from the `models` folder next to the app.

models/
  text/   multilingual sentence-embedding model (ONNX)   -> what a document is about, in 100+ languages
  clip/   image/text model (ONNX)                        -> what a picture or video shows
  llm/    optional small instruction model (.gguf)       -> nicer folder names
OCR uses the models bundled inside the `rapidocr_onnxruntime` package.

When a model is missing the app still works with simpler built-in methods (TF-IDF for text, colour/shape
features for images) so a run never fails because of a missing model.
"""

from __future__ import annotations

import json
import logging
import pickle
import re
import threading
import time
from pathlib import Path

import numpy as np

from .labels import LABEL_PROMPTS
from .textutil import STOPWORDS, strip_accents

log = logging.getLogger(__name__)

MANIFEST = "filesorter_model.json"


def _normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, 1e-12)


def _read_manifest(folder: Path) -> dict | None:
    f = folder / MANIFEST
    if f.is_file():
        try:
            return json.loads(f.read_text(encoding="utf-8"))
        except ValueError:
            log.warning("Invalid model manifest: %s", f)
    return None


def _ort_session(path: Path, threads: int):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = max(1, threads)
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.log_severity_level = 3
    return ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])


def _pad_id(tok, default: int = 0) -> int:
    for t in ("<pad>", "[PAD]", "<|endoftext|>", "</s>"):
        i = tok.token_to_id(t)
        if i is not None:
            return i
    return default


# ---------------------------------------------------------------------------------------------------------------
# Text embeddings
# ---------------------------------------------------------------------------------------------------------------
class OnnxTextEmbedder:
    needs_fit = False

    def __init__(self, folder: Path, threads: int):
        from tokenizers import Tokenizer

        m = _read_manifest(folder) or {}
        onnx_file = folder / m.get("onnx", "model.onnx")
        if not onnx_file.is_file():
            found = sorted(folder.rglob("*.onnx"))
            if not found:
                raise FileNotFoundError(f"no .onnx file in {folder}")
            onnx_file = found[0]
        self.name = m.get("name", folder.name)
        self.space = f"text:{self.name}"
        self.prefix = m.get("prefix", "")
        self.pooling = m.get("pooling", "mean")
        self.max_length = int(m.get("max_length", 512))
        self.chunk_chars = int(m.get("chunk_chars", 1500))
        self.tok = Tokenizer.from_file(str(folder / m.get("tokenizer", "tokenizer.json")))
        self.tok.no_padding()
        self.tok.enable_truncation(self.max_length)
        self.pad = int(m.get("pad_id", _pad_id(self.tok, 1)))
        self.session = _ort_session(onnx_file, threads)
        self.inputs = {i.name for i in self.session.get_inputs()}
        outs = self.session.get_outputs()
        self.output = next((o.name for o in outs if o.name in ("sentence_embedding", "text_embeds")), None)
        if self.output is None:
            self.output = next((o.name for o in outs if o.name == "last_hidden_state"), outs[0].name)
        self.lock = threading.Lock()

    def fit(self, texts):  # not needed for neural models
        pass

    def _encode(self, texts: list[str], batch: int = 16) -> np.ndarray:
        encs = self.tok.encode_batch([self.prefix + t for t in texts])
        order = np.argsort([len(e.ids) for e in encs])
        out = np.zeros((len(texts), 0), dtype=np.float32)
        result: list[np.ndarray | None] = [None] * len(texts)
        for s in range(0, len(order), batch):
            idx = order[s:s + batch]
            L = max(len(encs[i].ids) for i in idx)
            ids = np.full((len(idx), L), self.pad, dtype=np.int64)
            mask = np.zeros((len(idx), L), dtype=np.int64)
            for r, i in enumerate(idx):
                e = encs[i].ids
                ids[r, :len(e)] = e
                mask[r, :len(e)] = 1
            feeds = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self.inputs:
                feeds["token_type_ids"] = np.zeros_like(ids)
            feeds = {k: v for k, v in feeds.items() if k in self.inputs}
            with self.lock:
                y = self.session.run([self.output], feeds)[0]
            if y.ndim == 3:
                if self.pooling == "cls":
                    y = y[:, 0]
                else:
                    m = mask[..., None].astype(np.float32)
                    y = (y * m).sum(1) / np.maximum(m.sum(1), 1)
            for r, i in enumerate(idx):
                result[i] = y[r]
        if not texts:
            return out
        return _normalize(np.stack(result))

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embeds documents. Long texts are represented by their beginning and middle."""
        chunks, owner = [], []
        for i, t in enumerate(texts):
            t = t or ""
            parts = [t[: self.chunk_chars]]
            if len(t) > self.chunk_chars * 3:
                mid = len(t) // 2
                parts.append(t[mid: mid + self.chunk_chars])
            for p in parts:
                chunks.append(p)
                owner.append(i)
        if not chunks:
            return np.zeros((0, 1), dtype=np.float32)
        vecs = self._encode(chunks)
        out = np.zeros((len(texts), vecs.shape[1]), dtype=np.float32)
        np.add.at(out, np.asarray(owner), vecs)
        return _normalize(out)

    def save(self, path: Path):
        pass

    def load(self, path: Path) -> bool:
        return True


class TfidfTextEmbedder:
    """Built-in fallback (no model files): TF-IDF + LSA. Must be fitted on the collection first."""

    needs_fit = True
    name = "built-in TF-IDF"

    def __init__(self):
        self.vec = None
        self.svd = None
        self.fit_id = 0

    @property
    def space(self) -> str:
        return f"text:tfidf-lsa:{self.fit_id}"

    def oov_ratio(self, texts: list[str]) -> float:
        """Share of words in `texts` that the fitted vocabulary does not know."""
        if self.vec is None:
            return 1.0
        analyze = self.vec.build_analyzer()
        vocab = self.vec.vocabulary_
        toks = [t for txt in texts for t in analyze(txt) if " " not in t]
        return sum(t not in vocab for t in toks) / len(toks) if toks else 0.0

    def fit(self, texts: list[str]):
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        n = len(texts)
        stop = sorted(STOPWORDS | {strip_accents(w) for w in STOPWORDS})
        self.vec = TfidfVectorizer(
            lowercase=True, token_pattern=r"(?u)\b[^\W\d_]{3,}\b", stop_words=stop,
            ngram_range=(1, 2), sublinear_tf=True, min_df=2 if n >= 40 else 1, max_df=0.6 if n >= 20 else 1.0,
            max_features=60000, strip_accents="unicode",
        )
        try:
            X = self.vec.fit_transform(texts)
        except ValueError:  # empty vocabulary
            self.vec = TfidfVectorizer(lowercase=True, token_pattern=r"(?u)\b\w+\b")
            X = self.vec.fit_transform([t or "empty" for t in texts])
        k = min(256, X.shape[0] - 1, X.shape[1] - 1)
        self.svd = TruncatedSVD(n_components=k, random_state=0).fit(X) if k >= 2 else None
        self.fit_id = time.time_ns()

    def embed(self, texts: list[str]) -> np.ndarray:
        if self.vec is None:
            raise RuntimeError("TF-IDF embedder used before fit()")
        X = self.vec.transform(texts)
        Y = self.svd.transform(X) if self.svd is not None else X.toarray()
        return _normalize(Y)

    def save(self, path: Path):
        with open(path, "wb") as f:
            pickle.dump({"vec": self.vec, "svd": self.svd, "fit_id": self.fit_id}, f)

    def load(self, path: Path) -> bool:
        try:
            with open(path, "rb") as f:
                d = pickle.load(f)  # our own file, written by save()
            self.vec, self.svd, self.fit_id = d["vec"], d["svd"], d.get("fit_id", 0)
            return True
        except (OSError, pickle.UnpicklingError, KeyError, EOFError):
            return False


# ---------------------------------------------------------------------------------------------------------------
# Visual embeddings
# ---------------------------------------------------------------------------------------------------------------
class ClipModel:
    has_labels = True

    def __init__(self, folder: Path, threads: int):
        from tokenizers import Tokenizer

        m = _read_manifest(folder) or {}
        self.name = m.get("name", folder.name)
        self.space = f"visual:{self.name}"
        self.size = int(m.get("image_size", 224))
        self.mean = np.array(m.get("mean", [0.48145466, 0.4578275, 0.40821073]), dtype=np.float32)[:, None, None]
        self.std = np.array(m.get("std", [0.26862954, 0.26130258, 0.27577711]), dtype=np.float32)[:, None, None]
        self.vision = _ort_session(folder / m.get("vision_onnx", "vision_model.onnx"), threads)
        self.text = _ort_session(folder / m.get("text_onnx", "text_model.onnx"), threads)
        self.tok = Tokenizer.from_file(str(folder / m.get("tokenizer", "tokenizer.json")))
        self.tok.no_padding()
        self.ctx = int(m.get("context_length", 77))
        self.tok.enable_truncation(self.ctx)
        self.pad = int(m.get("pad_id", _pad_id(self.tok, 0)))
        self.v_in = self.vision.get_inputs()[0].name
        self.v_out = self._pick(self.vision, "image_embeds")
        self.t_inputs = {i.name for i in self.text.get_inputs()}
        self.t_out = self._pick(self.text, "text_embeds")
        self.lock = threading.Lock()
        self._labels: np.ndarray | None = None

    @staticmethod
    def _pick(sess, preferred: str) -> str:
        outs = sess.get_outputs()
        for o in outs:
            if o.name == preferred:
                return o.name
        for o in outs:
            if len(o.shape) == 2:
                return o.name
        return outs[0].name

    def _prep(self, img) -> np.ndarray:
        from PIL import Image

        img = img.convert("RGB")
        w, h = img.size
        s = self.size / min(w, h)
        img = img.resize((max(self.size, round(w * s)), max(self.size, round(h * s))), Image.Resampling.BICUBIC)
        w, h = img.size
        left, top = (w - self.size) // 2, (h - self.size) // 2
        img = img.crop((left, top, left + self.size, top + self.size))
        a = np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
        return (a - self.mean) / self.std

    def embed_images(self, images: list) -> np.ndarray:
        if not images:
            return np.zeros((0, 1), dtype=np.float32)
        out = []
        for s in range(0, len(images), 16):
            batch = np.stack([self._prep(im) for im in images[s:s + 16]]).astype(np.float32)
            with self.lock:
                out.append(self.vision.run([self.v_out], {self.v_in: batch})[0])
        return _normalize(np.concatenate(out))

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        vecs = []
        for t in texts:
            ids = self.tok.encode(t).ids[: self.ctx]
            arr = np.full((1, self.ctx), self.pad, dtype=np.int64)
            arr[0, :len(ids)] = ids
            mask = np.zeros((1, self.ctx), dtype=np.int64)
            mask[0, :len(ids)] = 1
            feeds = {"input_ids": arr, "attention_mask": mask}
            feeds = {k: v for k, v in feeds.items() if k in self.t_inputs}
            with self.lock:
                vecs.append(self.text.run([self.t_out], feeds)[0][0])
        return _normalize(np.stack(vecs))

    def label_embeddings(self) -> np.ndarray:
        if self._labels is None:
            self._labels = self.embed_texts(LABEL_PROMPTS)
        return self._labels


class BasicVisualEmbedder:
    """Built-in fallback: colour histogram + tiny thumbnail. Groups look-alike pictures but can't name them."""

    has_labels = False
    space = "visual:basic"
    name = "built-in colour/shape features"

    def embed_images(self, images: list) -> np.ndarray:
        from PIL import Image

        feats = []
        for im in images:
            hsv = np.asarray(im.convert("RGB").resize((64, 64)).convert("HSV"), dtype=np.float32)
            hist, _ = np.histogramdd(hsv.reshape(-1, 3), bins=(8, 4, 4), range=((0, 256), (0, 256), (0, 256)))
            hist = np.sqrt(hist.ravel() / max(hist.sum(), 1))
            g = np.asarray(im.convert("L").resize((12, 12), Image.Resampling.BILINEAR), dtype=np.float32).ravel()
            g = (g - g.mean()) / (g.std() + 1e-6) * 0.08
            feats.append(np.concatenate([hist, g]))
        if not feats:
            return np.zeros((0, 1), dtype=np.float32)
        return _normalize(np.stack(feats))

    def label_embeddings(self):
        return None


# ---------------------------------------------------------------------------------------------------------------
# OCR and the naming language model (both optional, loaded lazily)
# ---------------------------------------------------------------------------------------------------------------
class OcrEngine:
    def __init__(self):
        from rapidocr_onnxruntime import RapidOCR  # models ship inside the package

        self.engine = RapidOCR()
        self.lock = threading.Lock()

    def read(self, img, min_conf: float = 0.6) -> str:
        im = img.convert("RGB")
        if max(im.size) > 1600:
            im = im.copy()
            im.thumbnail((1600, 1600))
        arr = np.asarray(im)[:, :, ::-1].copy()  # RGB -> BGR (OpenCV convention)
        with self.lock:
            result, _ = self.engine(arr)
        if not result:
            return ""
        return " ".join(r[1] for r in result if len(r) > 2 and float(r[2]) >= min_conf)


class LocalNamer:
    """Turns keywords + example titles into a short, specific folder name with a local .gguf model."""

    SYSTEM = (
        "You name folders for a file organiser. Reply with ONLY the folder name: 1 to 5 words, specific to the "
        "subject (for example 'Q3 sales forecasts', 'Onboarding new hires', 'Logo redesign Acme', "
        "'Notes de frais', 'Formation Excel'). No dates, no quotes, no explanation. "
        "Write it in the language most of the examples are written in."
    )

    def __init__(self, gguf: Path, threads: int):
        from llama_cpp import Llama

        self.llm = Llama(model_path=str(gguf), n_ctx=2048, n_threads=max(1, threads), verbose=False, seed=7)
        self.name = gguf.stem

    def _ask(self, prompt: str) -> str | None:
        try:
            r = self.llm.create_chat_completion(
                messages=[{"role": "system", "content": self.SYSTEM}, {"role": "user", "content": prompt}],
                temperature=0.1, max_tokens=24,
            )
            text = r["choices"][0]["message"]["content"] or ""
        except Exception as exc:  # noqa: BLE001
            log.warning("Naming model failed: %s", exc)
            return None
        return clean_llm_name(text)

    def name_topic(self, keywords: list[str], examples: list[str]) -> str | None:
        ex = "\n".join(f"- {e[:120]}" for e in examples[:10])
        return self._ask(f"Keywords: {', '.join(keywords[:12])}\nExample files:\n{ex}\nFolder name:")

    def name_area(self, child_names: list[str], keywords: list[str], avoid: list[str] | None = None
                  ) -> str | None:
        subs = "\n".join(f"- {c}" for c in child_names[:15])
        rule = f"\nDo NOT answer any of: {', '.join(avoid)}." if avoid else ""
        return self._ask(
            "Give the name of the broad parent folder that contains these sub-folders: 1 to 3 words, broader "
            "than the sub-folders, like a field or a department (for example 'Finance', 'Human resources', "
            "'IT & security', 'Marketing', 'Training', 'Research', 'Legal', 'Sales', 'Operations'). "
            f"It must be different from the sub-folder names.{rule}\n"
            f"Sub-folders:\n{subs}\nKeywords: {', '.join(keywords[:10])}\nParent folder name:"
        )

    def close(self):
        try:
            self.llm.close()
        except Exception:  # noqa: BLE001
            pass


def clean_llm_name(text: str) -> str | None:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()
    line = text.splitlines()[0] if text else ""
    line = re.sub(r"^(folder name|parent folder name|nom du dossier)\s*:\s*", "", line, flags=re.I)
    line = line.strip().strip("\"'`*#.-: ").strip()
    if " " not in line:
        # 'VehicleRoutingStudy' -> 'Vehicle Routing Study' (but leave 'iPhone', 'VBA', 'PowerBI' alone)
        parts = re.split(r"(?<=[a-zà-ÿ])(?=[A-ZÀ-Þ][a-zà-ÿ])", line)
        if len(parts) >= 3 or (len(parts) == 2 and min(map(len, parts)) >= 4):
            line = " ".join(parts)
    if not line or len(line) > 60 or len(line.split()) > 7:
        return None
    return line


# ---------------------------------------------------------------------------------------------------------------
class ModelHub:
    """Finds and loads the local models. Missing models are replaced by built-in fallbacks."""

    def __init__(self, models_dir: Path, threads: int = 2, ocr: bool = True, ai_naming: bool = True):
        self.dir = Path(models_dir)
        self.threads = threads
        self.want_ocr = ocr
        self.want_llm = ai_naming
        self.problems: list[str] = []
        self.text = self._load_text()
        self.visual = self._load_visual()
        self._ocr = None
        self._ocr_failed = False
        self._namer = None
        self._namer_failed = False

    def _load_text(self):
        folder = self.dir / "text"
        if folder.is_dir():
            try:
                return OnnxTextEmbedder(folder, self.threads)
            except Exception as exc:  # noqa: BLE001
                self.problems.append(f"text model could not be loaded ({exc}); using built-in TF-IDF")
                log.exception("text model")
        return TfidfTextEmbedder()

    def _load_visual(self):
        folder = self.dir / "clip"
        if folder.is_dir():
            try:
                return ClipModel(folder, self.threads)
            except Exception as exc:  # noqa: BLE001
                self.problems.append(f"image model could not be loaded ({exc}); using basic image features")
                log.exception("clip model")
        return BasicVisualEmbedder()

    @property
    def ocr(self) -> OcrEngine | None:
        if not self.want_ocr or self._ocr_failed:
            return None
        if self._ocr is None:
            try:
                self._ocr = OcrEngine()
            except Exception as exc:  # noqa: BLE001
                self._ocr_failed = True
                self.problems.append(f"OCR unavailable ({exc.__class__.__name__})")
                return None
        return self._ocr

    def llm_file(self) -> Path | None:
        folder = self.dir / "llm"
        if not folder.is_dir():
            return None
        files = sorted(folder.glob("*.gguf"), key=lambda p: p.stat().st_size)
        return files[-1] if files else None

    @property
    def namer(self) -> LocalNamer | None:
        if not self.want_llm or self._namer_failed:
            return None
        if self._namer is None:
            f = self.llm_file()
            if f is None:
                self._namer_failed = True
                return None
            try:
                self._namer = LocalNamer(f, self.threads)
            except Exception as exc:  # noqa: BLE001
                self._namer_failed = True
                self.problems.append(f"naming model unavailable ({exc.__class__.__name__})")
                return None
        return self._namer

    def release_namer(self):
        if self._namer is not None:
            self._namer.close()
            self._namer = None

    def status(self) -> dict:
        ocr_ok = False
        if self.want_ocr:
            try:
                import rapidocr_onnxruntime  # noqa: F401

                ocr_ok = True
            except Exception:  # noqa: BLE001
                ocr_ok = False
        llm = self.llm_file()
        try:
            import llama_cpp  # noqa: F401

            llm_runtime = True
        except Exception:  # noqa: BLE001
            llm_runtime = False
        return {
            "text_model": self.text.name,
            "text_model_ok": not isinstance(self.text, TfidfTextEmbedder),
            "image_model": self.visual.name,
            "image_model_ok": isinstance(self.visual, ClipModel),
            "ocr": ocr_ok,
            "naming_model": llm.stem if (llm and llm_runtime) else None,
            "problems": list(self.problems),
        }
