"""The sorting engine: scan -> read content -> find duplicates -> organise -> copy -> report.

Source files are never modified, moved or deleted: everything is *copied* into the destination folder.
The first run creates the folder structure; later runs (manual or in the background) only process new or
changed files and slot them into the existing folders, creating new folders when a new subject appears.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path

import numpy as np

from . import offline
from .config import Settings
from .extract import KIND_FOLDERS, Extracted, extract
from .features import (colour_signature, full_hash, hamming_matrix, head_tail_hash, image_hashes,
                       text_fingerprint, to_u64)
from .labels import LABELS, PHOTOS, TEXTUAL_AREAS, pick_label
from .models import ModelHub
from .organize import (CONFIDENT, Levels, Terms, VisualItem, acceptable_match, area_threshold, event_name,
                       keyword_table, organize_documents, organize_visual, segment_events, visual_area)
from .paths import (DUPLICATE_DIRNAME, IS_WINDOWS, STATE_DIRNAME, UNSORTED_DIRNAME, is_relative_to, long_path,
                    models_dir)
from .state import ANALYSIS_VERSION, State, from_blob, to_blob
from .textutil import GENERIC_FOLDERS, keywords, looks_like_copy, safe_name, split_name, unique_name

log = logging.getLogger(__name__)

# Windows file attributes
_HIDDEN, _SYSTEM = 0x2, 0x4
_CLOUD_ONLY = 0x1000 | 0x40000 | 0x400000      # OFFLINE | RECALL_ON_OPEN | RECALL_ON_DATA_ACCESS
_FREE_SPACE_RESERVE = 2 * 1024 ** 3            # keep 2 GB free on the destination drive
BATCH = 32


class Cancelled(Exception):
    pass


@dataclass
class Entry:
    path: Path
    root: Path
    size: int
    mtime: float


def _system_dirs() -> list[str]:
    if not IS_WINDOWS:
        return ["/proc", "/sys", "/dev", "/usr", "/bin", "/sbin", "/lib", "/etc", "/var/lib", "/snap"]
    env = os.environ
    dirs = [env.get("SystemRoot", r"C:\Windows"), env.get("ProgramFiles", r"C:\Program Files"),
            env.get("ProgramFiles(x86)", r"C:\Program Files (x86)"), env.get("ProgramData", r"C:\ProgramData"),
            env.get("APPDATA", ""), env.get("LOCALAPPDATA", "")]
    return [os.path.normcase(os.path.abspath(d)) for d in dirs if d]


def path_context(rel: str) -> str:
    """Meaningful folder names + file name, e.g. 'Formation Excel 2023 Module 2 exercice tableaux'.
    Boilerplate names ('Screenshot 2024-05-02 101500', "Capture d'écran…", 'IMG_4100', 'Scan_001') say nothing
    about the subject and would dominate short texts, so they are left out."""
    parts = Path(rel).parts
    if not parts:
        return ""
    folders = [split_name(p) for p in parts[:-1][-3:] if p.lower().strip() not in GENERIC_FOLDERS]
    name = split_name(parts[-1])
    if not keywords(name):
        name = ""
    return " ".join(x for x in folders + [name] if x)


class Engine:
    def __init__(self, settings: Settings, hub: ModelHub | None = None, progress=None,
                 cancel: threading.Event | None = None, pause: threading.Event | None = None,
                 models: Path | None = None):
        offline.enable()
        self.s = settings
        self.dest = Path(settings.destination).expanduser()
        self.progress = progress or (lambda stage, done, total, msg="": None)
        self.cancel = cancel or threading.Event()
        self.pause = pause or threading.Event()
        self._hub = hub
        self._models = models or models_dir()
        self.state: State | None = None
        self.run_id = 0
        self.stats: Counter = Counter()
        self.skipped: Counter = Counter()
        self.errors: list[str] = []

    # ------------------------------------------------------------------------------------------------------------
    @property
    def hub(self) -> ModelHub:
        if self._hub is None:
            self.progress("Loading local models", 0, 0, "")
            self._hub = ModelHub(self._models, self.s.worker_threads(), self.s.ocr, self.s.ai_naming)
        return self._hub

    def _checkpoint(self):
        while self.pause.is_set() and not self.cancel.is_set():
            time.sleep(0.3)
        if self.cancel.is_set():
            raise Cancelled()

    def validate(self) -> list[str]:
        problems = []
        if not self.s.sources:
            problems.append("Choose at least one folder to sort.")
        if not self.s.destination:
            problems.append("Choose a destination folder.")
        for src in self.s.sources:
            p = Path(src)
            if not p.is_dir():
                problems.append(f"Folder not found: {src}")
            elif self.s.destination and (p.resolve() == self.dest.resolve() or is_relative_to(p, self.dest)):
                problems.append(f"The destination cannot contain a folder to sort: {src}")
        return problems

    # ------------------------------------------------------------------------------------------------------------
    def run(self) -> dict:
        problems = self.validate()
        if problems:
            raise ValueError("\n".join(problems))
        t0 = time.time()
        self.dest.mkdir(parents=True, exist_ok=True)
        self.state = st = State(self.dest / STATE_DIRNAME)
        first = not st.categories() and not st.placements()
        self.run_id = st.start_run("first" if first else "update")
        cancelled = False
        try:
            entries = self.scan()
            self.analyze(entries)
            self._checkpoint()
            current = st.files("seen_run=?", (self.run_id,))
            new_ids = {r["id"] for r in current if st.placement(r["id"]) is None}
            dupes = self.find_duplicates(current, None if first else new_ids)
            self._checkpoint()
            if first:
                self.organize_all(current, dupes)
            elif new_ids:
                self.assign_new(current, new_ids, dupes)
            st.commit()
            self.mark_changed_sources(current)
            self.copy_pending()
        except Cancelled:
            cancelled = True
            log.info("Run cancelled")
        except Exception:
            st.db.rollback()   # never keep a half-built folder structure; analysis already saved stays
            raise
        finally:
            if self._hub is not None:
                self._hub.release_namer()
            self.stats["seconds"] = round(time.time() - t0, 1)
            self.stats["cancelled"] = int(cancelled)
            stats = dict(self.stats)
            stats["skipped"] = dict(self.skipped)
            st.commit()
            try:
                from .report import write_report

                self.progress("Writing report", 0, 0, "")
                stats["report"] = str(write_report(st, self.dest, stats, self.hub_status(), self.s))
            except Exception:  # noqa: BLE001
                log.exception("report")
            st.finish_run(self.run_id, stats)
            st.close()
        self.progress("Done", 1, 1, "")
        return stats

    def hub_status(self) -> dict:
        status = self._hub.status() if self._hub is not None else {}
        status["offline"] = offline.is_enabled()
        return status

    # ------------------------------------------------------------------------------------------------------------
    # 1. Scan
    # ------------------------------------------------------------------------------------------------------------
    def scan(self) -> list[Entry]:
        excludes = [p.lower() for p in self.s.exclude]
        sysdirs = _system_dirs()
        dest = self.dest.resolve()
        roots = sorted({Path(s).resolve() for s in self.s.sources}, key=lambda p: len(str(p)))
        roots = [r for i, r in enumerate(roots) if not any(is_relative_to(r, o) for o in roots[:i])]
        max_bytes = self.s.max_file_mb * 1024 * 1024 if self.s.max_file_mb else 0
        entries: list[Entry] = []
        for root in roots:
            stack = [root]
            while stack:
                if len(entries) % 200 == 0:
                    self._checkpoint()
                    self.progress("Scanning folders", len(entries), 0, str(stack[-1]))
                d = stack.pop()
                try:
                    it = os.scandir(d)
                except OSError:
                    self.skipped["folders without access"] += 1
                    continue
                with it:
                    for e in it:
                        low = e.name.lower()
                        if any(fnmatch(low, p) for p in excludes):
                            self.skipped["excluded by pattern"] += 1
                            continue
                        try:
                            if e.is_symlink() or (hasattr(e, "is_junction") and e.is_junction()):
                                continue
                            st = e.stat(follow_symlinks=False)
                            attrs = getattr(st, "st_file_attributes", 0)
                            hidden = bool(attrs & (_HIDDEN | _SYSTEM)) or e.name.startswith(".")
                            if e.is_dir(follow_symlinks=False):
                                p = Path(e.path)
                                if p == dest or is_relative_to(p, dest):
                                    continue
                                if hidden and not self.s.include_hidden:
                                    continue
                                if os.path.normcase(str(p)) in sysdirs:
                                    continue
                                stack.append(p)
                            elif e.is_file(follow_symlinks=False):
                                if attrs & _CLOUD_ONLY and self.s.skip_cloud_only:
                                    self.skipped["online-only (cloud) files"] += 1
                                elif hidden and not self.s.include_hidden:
                                    self.skipped["hidden or system files"] += 1
                                elif st.st_size == 0:
                                    self.skipped["empty files"] += 1
                                elif max_bytes and st.st_size > max_bytes:
                                    self.skipped["larger than the size limit"] += 1
                                else:
                                    entries.append(Entry(Path(e.path), root, st.st_size, st.st_mtime))
                        except OSError:
                            self.skipped["unreadable entries"] += 1
        self.stats["files found"] = len(entries)
        self.stats["bytes found"] = sum(e.size for e in entries)
        return entries

    # ------------------------------------------------------------------------------------------------------------
    # 2. Read content (only new or changed files)
    # ------------------------------------------------------------------------------------------------------------
    def analyze(self, entries: list[Entry]) -> list[Entry]:
        st = self.state
        known = st.known_files()
        todo, unchanged = [], []
        for e in entries:
            k = known.get(str(e.path))
            if k is not None and k["size"] == e.size and abs((k["mtime"] or 0) - e.mtime) < 1e-3 \
                    and (k["analyzed"] or 0) >= ANALYSIS_VERSION:
                unchanged.append(k["id"])
            else:
                todo.append(e)
        st.db.executemany("UPDATE files SET seen_run=? WHERE id=?", [(self.run_id, i) for i in unchanged])
        st.commit()
        self.stats["new or changed files"] = len(todo)
        if not todo:
            return todo
        hub = self.hub
        total = len(todo)
        done = 0
        with ThreadPoolExecutor(max_workers=self.s.worker_threads()) as pool:
            for s in range(0, total, BATCH):
                self._checkpoint()
                batch = todo[s:s + BATCH]
                read = list(pool.map(self._read, batch))
                self._understand(read, hub)
                for rec, _ex in read:
                    rec["seen_run"] = self.run_id
                    rec["analyzed"] = ANALYSIS_VERSION
                    st.upsert_file(rec)
                st.commit()
                done += len(batch)
                self.progress("Reading files", done, total, batch[-1].path.name)
        return todo

    def _read(self, e: Entry) -> tuple[dict, Extracted]:
        ex = extract(e.path)
        try:
            rel = str(e.path.relative_to(e.root))
        except ValueError:
            rel = e.path.name
        meta = dict(ex.meta)
        rec = dict(path=str(e.path), root=str(e.root), rel=rel, name=e.path.name, ext=e.path.suffix.lower(),
                   kind=ex.kind, size=e.size, mtime=e.mtime, error=ex.error, title=ex.title,
                   taken=meta.get("taken") or (meta.get("created") if ex.kind == "video" else None),
                   duration=meta.get("duration"), head_hash=None, full_hash=None, phash=None, dhash=None,
                   detail=None, aspect=None, csig=None, vhashes=None, label=None, label_score=None, area_hint=None,
                   temb=None, tspace=None, vemb=None, vspace=None)
        if ex.images:
            try:
                if ex.kind == "video":
                    rec["vhashes"] = json.dumps([image_hashes(im)[0] for im in ex.images])
                else:
                    ph, dh, detail = image_hashes(ex.images[0])
                    rec.update(phash=ph, dhash=dh, detail=detail, csig=colour_signature(ex.images[0]))
                    w, h = meta.get("width"), meta.get("height")
                    if w and h:
                        rec["aspect"] = w / h
                    else:
                        iw, ih = ex.images[0].size
                        rec["aspect"] = iw / max(ih, 1)
            except Exception:  # noqa: BLE001
                log.debug("hash failed for %s", e.path, exc_info=True)
        if ex.kind not in ("image", "video", "audio"):
            rec["text_fp"] = text_fingerprint(ex.text)
        rec["meta"] = json.dumps(meta, default=str)
        return rec, ex

    def _understand(self, read: list[tuple[dict, Extracted]], hub: ModelHub) -> None:
        """Visual embeddings, labels, OCR, routing and text embeddings for one batch."""
        visual = hub.visual
        # ---- pictures: what do they show?
        vis_idx = [i for i, (_, ex) in enumerate(read) if ex.images]
        if vis_idx:
            flat, owner = [], []
            for i in vis_idx:
                for im in read[i][1].images:
                    flat.append(im)
                    owner.append(i)
            try:
                V = visual.embed_images(flat)
                per = defaultdict(list)
                for o, v in zip(owner, V):
                    per[o].append(v)
                L = visual.label_embeddings()
                for i, vs in per.items():
                    v = np.mean(vs, axis=0)
                    v = v / max(np.linalg.norm(v), 1e-12)
                    rec = read[i][0]
                    rec["vemb"], rec["vspace"] = to_blob(v), visual.space
                    if L is not None:
                        logits = 100.0 * (L @ v)
                        p = np.exp(logits - logits.max())
                        p /= p.sum()
                        camera = bool(json.loads(rec["meta"]).get("camera"))
                        j, score, area = pick_label(p, camera=camera)
                        rec.update(label=j, label_score=score, area_hint=area)
            except Exception as exc:  # noqa: BLE001
                log.exception("visual embedding failed")
                self.errors.append(f"image analysis failed: {exc}")

        # ---- OCR on pictures that probably contain text
        ocr = hub.ocr
        for rec, ex in read:
            if not ex.images or ocr is None:
                continue
            textual = rec.get("area_hint") in TEXTUAL_AREAS
            meta = json.loads(rec["meta"])
            if not (textual or ex.needs_ocr) or (meta.get("camera") and not textual):
                continue
            try:
                frames = ex.images if ex.kind != "video" else ex.images[1:3]
                found = " ".join(ocr.read(im) for im in frames)
                if found.strip():
                    ex.text = f"{ex.text} {found}".strip()
                    meta["ocr_chars"] = len(found)
                    rec["meta"] = json.dumps(meta, default=str)
                    if not rec.get("text_fp") and ex.kind not in ("image", "video"):
                        rec["text_fp"] = text_fingerprint(ex.text)
            except Exception:  # noqa: BLE001
                log.debug("OCR failed for %s", rec["path"], exc_info=True)

        # ---- decide how each file is understood: by its text, by its look, or not at all
        for rec, ex in read:
            ctx = path_context(rec["rel"])
            body = ex.text
            embed_text = " ".join(x for x in (ctx, rec.get("title") or "", body) if x)
            rec["embed_text"] = embed_text[:8000]
            n_kw = len(set(keywords(embed_text)))
            n_body_kw = len(set(keywords(body)))
            kind = rec["kind"]
            if kind == "program":
                route = "none"
            elif kind in ("image", "video") or (kind == "design" and ex.images):
                if n_body_kw >= 8 and (rec.get("area_hint") in TEXTUAL_AREAS or not visual.has_labels):
                    route = "text"
                elif rec.get("vemb") is not None:
                    route = "visual"
                else:
                    route = "none"
            elif ex.error and n_body_kw < 2:
                route = "none"
            elif kind in ("audio", "archive", "other"):
                route = "text" if n_kw >= 3 else "none"
            elif n_kw >= 2:
                route = "text"
            elif rec.get("vemb") is not None:
                route = "visual"   # e.g. a scanned PDF without readable text
            else:
                route = "none"
            rec["route"] = route

        # ---- text embeddings (neural model: right away; built-in TF-IDF: later, once fitted on everything)
        if not hub.text.needs_fit:
            idx = [i for i, (rec, _) in enumerate(read) if rec["route"] == "text"]
            if idx:
                T = hub.text.embed([read[i][0]["embed_text"] for i in idx])
                for i, v in zip(idx, T):
                    read[i][0]["temb"], read[i][0]["tspace"] = to_blob(v), hub.text.space
        for _, ex in read:
            ex.images = []  # free memory

    # ------------------------------------------------------------------------------------------------------------
    # 3. Duplicates (by content, whatever the name)
    # ------------------------------------------------------------------------------------------------------------
    def find_duplicates(self, rows, new_ids: set[int] | None) -> dict[int, int]:
        """Returns {duplicate file id: original file id}. `new_ids` limits comparisons to new files."""
        st = self.state
        self.progress("Finding duplicates", 0, 0, "")
        parent = {r["id"]: r["id"] for r in rows}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        byid = {r["id"]: r for r in rows}
        hashes = {r["id"]: (r["head_hash"], r["full_hash"]) for r in rows}

        # identical bytes: group by size, then quick hash, then full hash
        by_size = defaultdict(list)
        for r in rows:
            by_size[r["size"]].append(r["id"])
        candidates = [ids for ids in by_size.values() if len(ids) > 1 and (new_ids is None or set(ids) & new_ids)]
        done = 0
        for ids in candidates:
            self._checkpoint()
            heads = defaultdict(list)
            for i in ids:
                h, _ = hashes[i]
                if h is None:
                    try:
                        h = head_tail_hash(byid[i]["path"], byid[i]["size"])
                        st.update_file(i, head_hash=h)
                        hashes[i] = (h, hashes[i][1])
                    except OSError:
                        continue
                heads[h].append(i)
            for same in heads.values():
                if len(same) < 2:
                    continue
                fulls = defaultdict(list)
                for i in same:
                    f = hashes[i][1]
                    if f is None:
                        try:
                            f = full_hash(byid[i]["path"])
                            st.update_file(i, full_hash=f)
                        except OSError:
                            continue
                    fulls[f].append(i)
                for g in fulls.values():
                    for i in g[1:]:
                        union(g[0], i)
            done += 1
            if done % 50 == 0:
                self.progress("Finding duplicates", done, len(candidates), "identical files")
        st.commit()

        # same text content (re-saved / renamed documents, the same e-mail saved twice...)
        by_fp = defaultdict(list)
        for r in rows:
            if r["text_fp"]:
                by_fp[r["text_fp"]].append(r["id"])
        for g in by_fp.values():
            for i in g[1:]:
                union(g[0], i)

        # look-alike pictures (resized, recompressed, renamed)
        imgs = [r for r in rows if r["phash"] is not None and (r["detail"] or 0) >= 4.0]
        if len(imgs) > 1:
            ids = np.array([r["id"] for r in imgs])
            P = to_u64([r["phash"] for r in imgs])
            D = to_u64([r["dhash"] for r in imgs])
            A = np.log(np.array([max(r["aspect"] or 1.0, 1e-3) for r in imgs]))
            CS = np.stack([np.frombuffer(r["csig"], dtype=np.uint8) if r["csig"] else np.zeros(48, np.uint8)
                           for r in imgs])
            query = np.arange(len(imgs)) if new_ids is None else np.flatnonzero(np.isin(ids, list(new_ids)))
            for s in range(0, len(query), 256):
                self._checkpoint()
                q = query[s:s + 256]
                hp = hamming_matrix(P[q], P)
                hd = hamming_matrix(D[q], D)
                close = (hp <= 6) & (hd <= 10) & (np.abs(A[q][:, None] - A[None, :]) < 0.04)
                qa, bb = np.nonzero(close)
                keep = q[qa] != bb
                qa, bb = qa[keep], bb[keep]
                if len(qa):
                    diff = np.abs(CS[q[qa]].astype(np.int16) - CS[bb].astype(np.int16))
                    same_colours = (diff.mean(1) <= 6) & (diff.max(1) <= 24)
                    for a, b in zip(qa[same_colours], bb[same_colours]):
                        union(int(ids[q[a]]), int(ids[b]))
                self.progress("Finding duplicates", s, len(query), "look-alike pictures")

        # same video (same length and same frames)
        vids = sorted((r for r in rows if r["vhashes"] and r["duration"]), key=lambda r: r["duration"])
        for i, a in enumerate(vids):
            for b in vids[i + 1:]:
                if b["duration"] - a["duration"] > max(1.0, 0.01 * a["duration"]):
                    break
                if new_ids is not None and a["id"] not in new_ids and b["id"] not in new_ids:
                    continue
                ha, hb = json.loads(a["vhashes"]), json.loads(b["vhashes"])
                if len(ha) == len(hb) and ha:
                    dist = hamming_matrix(to_u64(ha), to_u64(hb)).diagonal()
                    if np.median(dist) <= 8:
                        union(a["id"], b["id"])

        groups = defaultdict(list)
        for i in parent:
            groups[find(i)].append(i)
        result: dict[int, int] = {}
        for g in groups.values():
            if len(g) < 2:
                continue
            original = self._pick_original([byid[i] for i in g])
            for i in g:
                if i != original:
                    result[i] = original
        self.stats["duplicates found"] = len(result)
        return result

    def _pick_original(self, rows) -> int:
        placed = [r for r in rows if (p := self.state.placement(r["id"])) is not None and p["role"] != "duplicate"]
        if placed:
            return placed[0]["id"]

        def score(r):
            meta = json.loads(r["meta"] or "{}")
            pixels = (meta.get("width") or 0) * (meta.get("height") or 0)
            return (not looks_like_copy(r["name"]), pixels, r["size"], -(r["mtime"] or 0), -len(r["name"]))

        return max(rows, key=score)["id"]

    # ------------------------------------------------------------------------------------------------------------
    # 4a. First run: build the folder structure from all files
    # ------------------------------------------------------------------------------------------------------------
    def _text_matrix(self, rows) -> np.ndarray:
        hub = self.hub
        if hub.text.needs_fit:
            texts = [r["embed_text"] or "" for r in rows]
            if getattr(hub.text, "vec", None) is None:
                pkl = self.state.folder / "tfidf.pkl"
                if not hub.text.load(pkl):
                    hub.text.fit(texts)
                    hub.text.save(pkl)
            X = hub.text.embed(texts)
            for r, v in zip(rows, X):
                self.state.update_file(r["id"], temb=to_blob(v), tspace=hub.text.space)
            return X
        out = []
        missing = [i for i, r in enumerate(rows) if r["temb"] is None or r["tspace"] != hub.text.space]
        if missing:
            V = hub.text.embed([rows[i]["embed_text"] or "" for i in missing])
            fixed = dict(zip(missing, V))
            for i, v in fixed.items():
                self.state.update_file(rows[i]["id"], temb=to_blob(v), tspace=hub.text.space)
        else:
            fixed = {}
        for i, r in enumerate(rows):
            out.append(fixed[i] if i in fixed else from_blob(r["temb"]))
        return np.stack(out) if out else np.zeros((0, 1), dtype=np.float32)

    def _visual_items(self, rows) -> tuple[list[VisualItem], np.ndarray]:
        items = []
        for r in rows:
            meta = json.loads(r["meta"] or "{}")
            items.append(VisualItem(
                kind=r["kind"], taken=r["taken"], mtime=r["mtime"] or 0.0, parent=Path(r["rel"]).parent.name,
                screen=bool(meta.get("screen")), camera=bool(meta.get("camera")), label=r["label"],
                label_score=r["label_score"] or 0.0,
                area_scores={r["area_hint"]: 1.0} if r["area_hint"] else None))
        X = np.stack([from_blob(r["vemb"]) for r in rows]) if rows else np.zeros((0, 1), dtype=np.float32)
        return items, X

    def organize_all(self, rows, dupes: dict[int, int]) -> None:
        st, hub = self.state, self.hub
        originals = [r for r in rows if r["id"] not in dupes]
        text_rows = [r for r in originals if r["route"] == "text"]
        vis_rows = [r for r in originals if r["route"] == "visual"]
        none_rows = [r for r in originals if r["route"] not in ("text", "visual")]
        cat_of: dict[int, int] = {}
        used_folders: set[str] = set()

        # --- documents
        self.progress("Creating topic folders", 0, 0, f"{len(text_rows)} documents")
        if text_rows:
            X = self._text_matrix(text_rows)
            terms = Terms.build([f"{r['title'] or ''} {r['embed_text'] or ''}" for r in text_rows])
            titles = [r["title"] or split_name(r["name"]) for r in text_rows]
            texts = [f"{r['title'] or ''} {r['embed_text'] or ''}" for r in text_rows]
            topics, levels, sim = organize_documents(X, terms, titles, self.s.detail, hub.namer, texts=texts)
            st.set_meta("text_levels", {"space": hub.text.space, "base": levels.base, "near": levels.near,
                                        "sim": sim})
            placed = set()
            for t in topics:
                folder = f"{t.area}/{t.name}"
                cid = st.add_category(area=t.area, name=t.name, folder=folder, kind="topic", space=hub.text.space,
                                      centroid=to_blob(t.centroid), n=len(t.members),
                                      keywords=json.dumps(t.keywords[:10], ensure_ascii=False))
                used_folders.add(folder.lower())
                for i in t.members:
                    r = text_rows[i]
                    conf = t.scores.get(i, 0.0)
                    reason = "about: " + ", ".join(t.keywords[:4]) if conf >= CONFIDENT else \
                        "closest topic (weaker match): " + ", ".join(t.keywords[:3])
                    st.set_placement(r["id"], category_id=cid, role="sorted", reason=reason, confidence=conf,
                                     dest=None, status="pending", dup_of=None)
                    cat_of[r["id"]] = cid
                    placed.add(i)
            unplaced = [i for i in range(len(text_rows)) if i not in placed]
            if unplaced and topics:
                C = np.stack([t.centroid for t in topics])
                S = (X[unplaced] @ C.T - levels.base) / max(sim - levels.base, 1e-6)
            for n, i in enumerate(unplaced):
                reason, conf = "no other file is about the same subject", 0.0
                if topics:
                    b = int(S[n].argmax())
                    conf = float(max(0.0, S[n][b]))
                    reason += f" (closest folder: {topics[b].area}/{topics[b].name}, match {conf:.2f})"
                st.set_placement(text_rows[i]["id"], category_id=None, role="unsorted", dest=None,
                                 status="pending", reason=reason, confidence=conf, dup_of=None)
        text_areas = {c["area"].lower() for c in st.categories()}

        # --- pictures and videos
        self.progress("Grouping pictures and videos", 0, 0, f"{len(vis_rows)} files")
        if vis_rows:
            items, V = self._visual_items(vis_rows)
            for t in organize_visual(items, V, self.s.detail):
                area = t.area if t.area.lower() not in text_areas else f"{t.area} (pictures)"
                folder = f"{area}/{t.name}"
                if folder.lower() in used_folders:
                    folder = f"{area}/{unique_name(t.name, {f.split('/', 1)[1] for f in used_folders if f.startswith(area.lower() + '/')})}"
                used_folders.add(folder.lower())
                cid = st.add_category(area=area, name=folder.split("/", 1)[1], folder=folder, kind=t.kind,
                                      space=hub.visual.space, centroid=to_blob(t.centroid), n=len(t.members),
                                      label=t.label, t_start=t.t_start, t_end=t.t_end, keywords="[]")
                for i in t.members:
                    r = vis_rows[i]
                    st.set_placement(r["id"], category_id=cid, role="sorted", confidence=1.0, dup_of=None,
                                     reason=self._visual_reason(r, t.kind), dest=None, status="pending")
                    cat_of[r["id"]] = cid

        # --- files without usable content
        for r in none_rows:
            st.set_placement(r["id"], category_id=None, role="unsorted", dest=None, status="pending", dup_of=None,
                             confidence=0.0, reason=self._unsorted_reason(r))
        self._place_duplicates(rows, dupes)
        st.commit()

    def _visual_reason(self, r, kind: str) -> str:
        if kind == "event":
            return "photos taken at the same time/event"
        if r["label"] is not None:
            return f"looks like: {LABELS[r['label']][0]}"
        return "grouped by date / folder"

    @staticmethod
    def _unsorted_reason(r) -> str:
        if r["kind"] == "program":
            return "program or installer (no readable content)"
        if r["error"]:
            return f"could not be read: {r['error']}"
        return "no readable content to understand"

    def _place_duplicates(self, rows, dupes: dict[int, int]) -> None:
        st = self.state
        byid = {r["id"]: r for r in rows}
        for dup, orig in dupes.items():
            if st.placement(dup) is not None and st.placement(dup)["role"] != "duplicate":
                continue
            p = st.placement(orig)
            name = byid[orig]["name"] if orig in byid else "?"
            st.set_placement(dup, category_id=p["category_id"] if p else None, role="duplicate", dup_of=orig,
                             reason=f"same content as {name}", confidence=1.0, dest=None, status="pending")
        self.stats["duplicates"] = len(dupes)

    # ------------------------------------------------------------------------------------------------------------
    # 4b. Later runs: slot new files into the existing folders
    # ------------------------------------------------------------------------------------------------------------
    def assign_new(self, rows, new_ids: set[int], dupes: dict[int, int]) -> None:
        st = self.state
        new_rows = [r for r in rows if r["id"] in new_ids and r["id"] not in dupes]
        self.stats["new files sorted into existing folders"] = 0
        cats = st.categories()

        # ---- documents
        text_new = [r for r in new_rows if r["route"] == "text"]
        unsorted_text = [r for r in rows if r["route"] == "text" and r["id"] not in new_ids
                         and (p := st.placement(r["id"])) is not None and p["role"] == "unsorted"]
        topic_cats = [c for c in cats if c["kind"] == "topic"]
        pending: list = []
        if text_new or unsorted_text:
            self._refresh_fallback_vocabulary(text_new)
            self._ensure_text_space(topic_cats)
            topic_cats = [c for c in st.categories() if c["kind"] == "topic"]
            lv = st.get_meta("text_levels") or {}
            if text_new:
                X = self._text_matrix(text_new)
            else:
                X = np.zeros((0, 1), dtype=np.float32)
            weak: dict[int, tuple] = {}
            if topic_cats and len(X) and lv:
                levels = Levels(lv["base"], lv["near"])
                sim = lv["sim"]
                C = np.stack([from_blob(c["centroid"]) for c in topic_cats])
                S = (X @ C.T - levels.base) / max(sim - levels.base, 1e-6)
                for r, row, v in zip(text_new, S, X):
                    b = int(row.argmax())
                    if row[b] >= CONFIDENT:
                        self._assign_topic(r, topic_cats[b], float(row[b]), v)
                        continue
                    ok = acceptable_match(row)
                    if ok is not None:   # weaker match: decided after new topics are tried
                        weak[r["id"]] = (topic_cats[ok], float(max(0.0, row[ok])), v)
                    pending.append(r)
            else:
                pending.extend(text_new)
            self._new_topics(pending, unsorted_text, weak)

        # ---- pictures and videos
        vis_new = [r for r in new_rows if r["route"] == "visual"]
        if vis_new:
            self._assign_visual(vis_new)

        # ---- no usable content
        for r in new_rows:
            if r["route"] not in ("text", "visual"):
                st.set_placement(r["id"], category_id=None, role="unsorted", dest=None, status="pending",
                                 dup_of=None, confidence=0.0, reason=self._unsorted_reason(r))
        self._place_duplicates(rows, {d: o for d, o in dupes.items() if d in new_ids})
        st.commit()

    def _assign_topic(self, r, c, score: float, v: np.ndarray) -> None:
        st = self.state
        kw = json.loads(c["keywords"] or "[]")
        reason = ("about: " if score >= CONFIDENT else "closest topic (weaker match): ") + ", ".join(kw[:4])
        st.set_placement(r["id"], category_id=c["id"], role="sorted", dup_of=None, dest=None, status="pending",
                         confidence=float(min(1.0, score)), reason=reason)
        cur = next((x for x in st.categories() if x["id"] == c["id"]), c)
        n = cur["n"] or 1
        st.update_category(c["id"], centroid=to_blob((from_blob(cur["centroid"]) * n + v) / (n + 1)), n=n + 1)
        self.stats["new files sorted into existing folders"] += 1

    def _refresh_fallback_vocabulary(self, new_rows) -> None:
        """The built-in TF-IDF method only knows words seen when it was fitted: refit when new files bring
        many unknown words (a new subject). Folder centroids are then recomputed by _ensure_text_space."""
        hub, st = self.hub, self.state
        if not hub.text.needs_fit or not new_rows:
            return
        pkl = st.folder / "tfidf.pkl"
        if hub.text.vec is None:
            hub.text.load(pkl)
        if hub.text.vec is not None and hub.text.oov_ratio([r["embed_text"] or "" for r in new_rows]) <= 0.2:
            return
        hub.text.fit([r["embed_text"] or "" for r in st.files("route='text'")])
        hub.text.save(pkl)

    def _ensure_text_space(self, topic_cats) -> None:
        """If the text model changed since the folders were created, recompute folder centroids with it."""
        st, hub = self.state, self.hub
        lv = st.get_meta("text_levels") or {}
        if not topic_cats or lv.get("space") == hub.text.space:
            return
        log.info("Text model changed (%s -> %s): re-indexing topic folders", lv.get("space"), hub.text.space)
        members = defaultdict(list)
        for p in st.placements():
            if p["role"] == "sorted" and p["category_id"]:
                members[p["category_id"]].append(p["file_id"])
        all_rows = [r for r in st.files("route='text'")]
        if hub.text.needs_fit and not hub.text.load(st.folder / "tfidf.pkl"):
            hub.text.fit([r["embed_text"] or "" for r in all_rows])
            hub.text.save(st.folder / "tfidf.pkl")
        byid = {r["id"]: r for r in all_rows}
        for c in topic_cats:
            rows = [byid[i] for i in members.get(c["id"], []) if i in byid]
            if rows:
                X = self._text_matrix(rows)
                st.update_category(c["id"], centroid=to_blob(X.mean(0)), space=hub.text.space, n=len(rows))
        refreshed = [r for r in st.files("route='text' AND temb IS NOT NULL AND tspace=?", (hub.text.space,))]
        if len(refreshed) >= 3:
            from .organize import DETAIL, measure_levels

            X = np.stack([from_blob(r["temb"]) for r in refreshed])
            levels = measure_levels(X)
            sim = levels.at(DETAIL.get(self.s.detail, DETAIL["normal"])[0])
            st.set_meta("text_levels", {"space": hub.text.space, "base": levels.base, "near": levels.near,
                                        "sim": sim})

    def _new_topics(self, pending, previously_unsorted, weak: dict | None = None) -> None:
        """New files that fit no folder well: create new topic folders when several share a subject.
        `weak` holds files with a so-so match to an existing folder; they go there only if they don't form a
        new subject together with other new files."""
        st, hub = self.state, self.hub
        weak = weak or {}
        pool = list(pending) + list(previously_unsorted)
        created = set()
        if len(pool) >= 4:
            X = self._text_matrix(pool)
            lv = st.get_meta("text_levels") or {}
            levels = Levels(lv["base"], lv["near"]) if lv else None
            all_rows = st.files("route='text'")
            pool_ids = [r["id"] for r in pool]
            index = {r["id"]: i for i, r in enumerate(all_rows)}
            terms_all = Terms.build([f"{r['title'] or ''} {r['embed_text'] or ''}" for r in all_rows])
            titles = [r["title"] or split_name(r["name"]) for r in pool]
            texts = [f"{r['title'] or ''} {r['embed_text'] or ''}" for r in pool]
            pool_terms = Terms(terms_all.M[[index[i] for i in pool_ids]], terms_all.vocab)
            topics, _, _ = organize_documents(X, pool_terms, titles, self.s.detail,
                                              hub.namer, levels, texts=texts)
            if topics:
                # keywords relative to the whole collection
                kws = keyword_table(terms_all, [[index[pool_ids[i]] for i in t.members] for t in topics])
                areas = self._area_centroids()
                for t, kw in zip(topics, kws):
                    t.keywords = kw or t.keywords
                    area = self._closest_area(t.centroid, areas, levels, self.s.detail) or t.area
                    siblings = {c["name"] for c in st.categories() if c["area"] == area}
                    name = unique_name(t.name, siblings)
                    folder = f"{area}/{name}"
                    cid = st.add_category(area=area, name=name, folder=folder, kind="topic", space=hub.text.space,
                                          centroid=to_blob(t.centroid), n=len(t.members),
                                          keywords=json.dumps(t.keywords[:10], ensure_ascii=False))
                    for i in t.members:
                        r = pool[i]
                        old = st.placement(r["id"])
                        status = "move" if old is not None and old["status"] == "copied" else "pending"
                        st.set_placement(r["id"], category_id=cid, role="sorted", dup_of=None,
                                         dest=old["dest"] if status == "move" else None, status=status,
                                         confidence=t.scores.get(i, 0.0), reason="about: " + ", ".join(kw[:4]))
                        created.add(r["id"])
                        self._rehome_duplicates(r["id"])
                self.stats["new topic folders"] = len(topics)
        for r in pending:
            if r["id"] in created:
                continue
            if r["id"] in weak:
                c, score, v = weak[r["id"]]
                self._assign_topic(r, c, score, v)
            else:
                st.set_placement(r["id"], category_id=None, role="unsorted", dest=None, status="pending",
                                 dup_of=None, confidence=0.0, reason="no other file is about the same subject")

    def _rehome_duplicates(self, original_id: int) -> None:
        st = self.state
        orig = st.placement(original_id)
        for p in st.db.execute("SELECT * FROM placements WHERE dup_of=?", (original_id,)).fetchall():
            status = "move" if p["status"] == "copied" else "pending"
            st.set_placement(p["file_id"], category_id=orig["category_id"], status=status,
                             dest=p["dest"] if status == "move" else None)

    def _area_centroids(self) -> dict[str, np.ndarray]:
        acc = defaultdict(list)
        for c in self.state.categories():
            if c["kind"] == "topic" and c["centroid"] is not None:
                acc[c["area"]].append(from_blob(c["centroid"]))
        return {a: np.mean(v, axis=0) for a, v in acc.items()}

    @staticmethod
    def _closest_area(centroid, areas, levels, detail) -> str | None:
        """The existing area whose documents are related to the new topic's (same rule as on the first run)."""
        if not areas or centroid is None or levels is None:
            return None
        names = list(areas)
        s = np.stack([areas[n] for n in names]) @ centroid   # average similarity between their documents
        b = int(s.argmax())
        return names[b] if s[b] >= area_threshold(levels, detail) else None

    def _assign_visual(self, rows) -> None:
        st, hub = self.state, self.hub
        items, V = self._visual_items(rows)
        cats = st.categories()
        events = [c for c in cats if c["kind"] == "event"]
        pending_photos = []
        for k, (r, it) in enumerate(zip(rows, items)):
            area = visual_area(it)
            area_cats = [c for c in cats if c["area"].split(" (pictures)")[0] == area]
            target = None
            if area == PHOTOS and it.taken:
                for c in events:
                    if c["t_start"] - 6 * 3600 <= it.taken <= c["t_end"] + 6 * 3600:
                        target = c
                        st.update_category(c["id"], t_start=min(c["t_start"], it.taken),
                                           t_end=max(c["t_end"], it.taken))
                        break
                if target is None:
                    pending_photos.append(k)
                    continue
            else:
                key = self._visual_key(it, area)
                target = next((c for c in area_cats if c["label"] == key and c["kind"] == "visual"), None)
                if target is None:
                    area_name = area_cats[0]["area"] if area_cats else area
                    siblings = {c["name"] for c in cats if c["area"] == area_name}
                    name = unique_name(safe_name(key, 70), siblings)
                    cid = st.add_category(area=area_name, name=name, folder=f"{area_name}/{name}", kind="visual",
                                          space=hub.visual.space, centroid=to_blob(V[k]), n=0, label=key,
                                          keywords="[]")
                    cats = st.categories()
                    target = next(c for c in cats if c["id"] == cid)
            st.set_placement(r["id"], category_id=target["id"], role="sorted", confidence=1.0, dup_of=None,
                             reason=self._visual_reason(r, target["kind"]), dest=None, status="pending")
            self.stats["new files sorted into existing folders"] += 1
        if pending_photos:
            self._new_events([rows[k] for k in pending_photos], [items[k] for k in pending_photos])

    @staticmethod
    def _visual_key(it: VisualItem, area: str) -> str:
        from datetime import datetime

        if it.label is not None and (LABELS[it.label][2] == area or area in ("Screen recordings", "Videos")):
            return LABELS[it.label][1]
        if it.parent and it.parent.lower() not in GENERIC_FOLDERS:
            return it.parent
        return datetime.fromtimestamp(it.taken or it.mtime).strftime("%Y-%m")

    def _new_events(self, rows, items) -> None:
        from datetime import datetime

        st, hub = self.state, self.hub
        cats = st.categories()
        photo_area = next((c["area"] for c in cats if c["area"].split(" (pictures)")[0] == PHOTOS), PHOTOS)
        segs = segment_events([it.taken for it in items])
        for s in segs:
            if len(s) >= 3:
                name = event_name(items, s)
                siblings = {c["name"] for c in st.categories() if c["area"] == photo_area}
                name = unique_name(name, siblings)
                cid = st.add_category(area=photo_area, name=name, folder=f"{photo_area}/{name}", kind="event",
                                      space=hub.visual.space, n=len(s), t_start=min(items[i].taken for i in s),
                                      t_end=max(items[i].taken for i in s), keywords="[]", label="")
                self.stats["new event folders"] += 1
            else:
                month = datetime.fromtimestamp(items[s[0]].taken).strftime("%Y-%m")
                c = next((c for c in st.categories() if c["kind"] == "month" and c["label"] == month), None)
                if c is None:
                    name = f"{month} Various photos"
                    cid = st.add_category(area=photo_area, name=name, folder=f"{photo_area}/{name}", kind="month",
                                          space=hub.visual.space, n=0, label=month, keywords="[]")
                else:
                    cid = c["id"]
            for i in s:
                st.set_placement(rows[i]["id"], category_id=cid, role="sorted", confidence=1.0, dup_of=None,
                                 reason="photos taken at the same time/event", dest=None, status="pending")

    # ------------------------------------------------------------------------------------------------------------
    # 5. Copy
    # ------------------------------------------------------------------------------------------------------------
    def mark_changed_sources(self, rows) -> None:
        """A source file that changed after it was copied gets copied again (same place)."""
        st = self.state
        for r in rows:
            p = st.placement(r["id"])
            if p and p["status"] == "copied" and (p["copied_size"] != r["size"] or
                                                  abs((p["copied_mtime"] or 0) - (r["mtime"] or 0)) > 1e-3):
                st.set_placement(r["id"], status="update")
        st.commit()

    def _folder_for(self, p, row, cats: dict, rows_by_id: dict) -> str:
        if p["category_id"] and p["category_id"] in cats:
            folder = cats[p["category_id"]]["folder"]
        else:
            kind_row = row
            if p["role"] == "duplicate" and p["dup_of"] in rows_by_id:
                kind_row = rows_by_id[p["dup_of"]]
            folder = f"{UNSORTED_DIRNAME}/{KIND_FOLDERS.get(kind_row['kind'], 'Other files')}"
        if p["role"] == "duplicate":
            folder = f"{folder}/{DUPLICATE_DIRNAME}"
        return folder

    def copy_pending(self) -> None:
        st = self.state
        cats = {c["id"]: c for c in st.categories()}
        rows_by_id = {r["id"]: r for r in st.files()}
        todo = [p for p in st.placements() if p["status"] in ("pending", "failed", "update", "move", "no space")]
        todo = [p for p in todo if p["file_id"] in rows_by_id]
        total_bytes = sum(rows_by_id[p["file_id"]]["size"] for p in todo if p["status"] != "move")
        taken = st.taken_dests()
        done_bytes = 0
        copied = moved = failed = 0
        for n, p in enumerate(todo):
            self._checkpoint()
            row = rows_by_id[p["file_id"]]
            folder = self._folder_for(p, row, cats, rows_by_id)
            old_rel = p["dest"]
            if p["status"] == "update" and old_rel:
                rel = old_rel
            elif p["status"] == "move" and old_rel and Path(old_rel).parent.as_posix().lower() == folder.lower():
                st.set_placement(p["file_id"], status="copied")
                continue
            else:
                rel = self._unique_dest(folder, row["name"], taken)
            target = self.dest / rel
            try:
                os.makedirs(long_path(target.parent), exist_ok=True)
                if p["status"] == "move" and old_rel and (self.dest / old_rel).exists():
                    os.replace(long_path(self.dest / old_rel), long_path(target))
                    if old_rel.lower() in taken:
                        taken.discard(old_rel.lower())
                    moved += 1
                else:
                    free = shutil.disk_usage(self.dest).free
                    if free - row["size"] < _FREE_SPACE_RESERVE:
                        st.set_placement(p["file_id"], status="no space", dest=None if p["status"] != "update"
                                         else old_rel)
                        self.skipped["not copied: destination drive almost full"] += 1
                        taken.discard(rel.lower())
                        continue
                    tmp = target.with_name(target.name + ".fs-partial")
                    shutil.copy2(long_path(row["path"]), long_path(tmp))
                    os.replace(long_path(tmp), long_path(target))
                    copied += 1
                    done_bytes += row["size"]
                st.set_placement(p["file_id"], dest=rel, status="copied", copied_size=row["size"],
                                 copied_mtime=row["mtime"], copied_at=time.time(), run_id=self.run_id)
            except OSError as exc:
                failed += 1
                taken.discard(rel.lower())
                st.set_placement(p["file_id"], status="failed")   # retried on the next run
                self.errors.append(f"{row['path']}: {exc.strerror or exc}")
                log.warning("copy failed %s -> %s: %s", row["path"], target, exc)
            if n % 20 == 0:
                st.commit()
                self.progress("Copying files", done_bytes, max(total_bytes, 1), row["name"])
        st.commit()
        self.stats["files copied"] = copied
        self.stats["files moved inside destination"] = moved
        self.stats["copy failures"] = failed
        self.stats["bytes copied"] = done_bytes

    def _unique_dest(self, folder: str, filename: str, taken: set[str]) -> str:
        name = safe_name(filename, max_len=150, fallback="file")
        stem, dot, ext = name.rpartition(".")
        if not dot or not stem:
            stem, ext = name, ""
        ext = f".{ext}" if ext else ""
        rel = f"{folder}/{stem}{ext}"
        i = 2
        while rel.lower() in taken or (self.dest / rel).exists():
            rel = f"{folder}/{stem} ({i}){ext}"
            i += 1
        taken.add(rel.lower())
        return rel


def lower_priority(background: bool = False) -> None:
    """Run gently: below-normal CPU priority (and low I/O priority in background mode)."""
    try:
        if IS_WINDOWS:
            import ctypes

            k32 = ctypes.windll.kernel32
            handle = k32.GetCurrentProcess()
            k32.SetPriorityClass(handle, 0x00004000)  # BELOW_NORMAL_PRIORITY_CLASS
        else:
            os.nice(5)
    except Exception:  # noqa: BLE001
        pass


def estimate_copy_size(settings: Settings) -> tuple[int, int]:
    """(bytes to copy, free bytes on destination) for the GUI, without reading file contents."""
    total = 0
    for src in settings.sources:
        for dirpath, _dirs, files in os.walk(src):
            for f in files:
                try:
                    total += os.path.getsize(os.path.join(dirpath, f))
                except OSError:
                    pass
    dest = Path(settings.destination)
    while not dest.exists() and dest.parent != dest:
        dest = dest.parent
    try:
        free = shutil.disk_usage(dest).free
    except OSError:
        free = 0
    return total, free


__all__ = ["Engine", "Cancelled", "lower_priority", "estimate_copy_size", "path_context"]
