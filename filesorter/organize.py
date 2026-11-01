"""Turns content vectors into precise, named folders.

Documents (and pictures with enough readable text) are grouped by meaning, two levels deep:
    <Area>/<Precise topic>/            e.g.  "Finance/Notes de frais", "Training/Excel pivot tables"
Pictures and videos are grouped by what they show, and photos of events by date:
    Photos & events/2024-06-14 Bowling/       Screenshots/Spreadsheets/

All similarity thresholds are *relative* to the collection (measured on the data itself), so the same code
works with the neural model, the built-in TF-IDF fallback, and with any language.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from .labels import GRAPHICS, LABELS, PHOTOS, RECORDINGS, SCANS, SCREENSHOTS, VIDEOS
from .textutil import (GENERIC_FOLDERS, nice_case, safe_name, sentence_case, surface_form, terms_and_phrases,
                       unique_name)

log = logging.getLogger(__name__)

DETAIL = {
    #          linkage strictness, min files per topic, max files per topic
    "coarse": (0.38, 5, 400),
    "normal": (0.52, 3, 200),
    "fine": (0.64, 2, 100),
}
CONFIDENT = 0.85   # fraction of the linkage threshold (relative scale) for a confident match
LOOSE = 0.55       # weaker matches are still sorted but flagged in the report
DIRECT_LIMIT = 4000
OTHER_IMAGES = "Other images"


@dataclass
class Levels:
    """Similarity scale of a collection: `base` = typical unrelated pair, `near` = typical closest neighbour."""
    base: float
    near: float

    def at(self, alpha: float) -> float:
        return self.base + alpha * (self.near - self.base)


@dataclass
class Topic:
    members: list[int]                 # indices into the input arrays
    name: str = ""
    area: str = ""
    keywords: list[str] = field(default_factory=list)
    centroid: np.ndarray | None = None  # mean (not normalised) of member vectors
    kind: str = "topic"                 # topic | event | visual
    label: str = ""
    t_start: float | None = None
    t_end: float | None = None
    scores: dict[int, float] = field(default_factory=dict)   # member index -> match score (0..1)


# ---------------------------------------------------------------------------------------------------------------
# Generic clustering
# ---------------------------------------------------------------------------------------------------------------
def measure_levels(X: np.ndarray, sample: int = 2000, seed: int = 0) -> Levels:
    n = len(X)
    if n < 3:
        return Levels(0.0, 1.0)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=min(n, sample), replace=False)
    S = X[idx] @ X[idx].T
    iu = np.triu_indices(len(idx), 1)
    base = float(np.median(S[iu]))
    np.fill_diagonal(S, -np.inf)
    near = float(np.median(S.max(axis=1)))
    if near - base < 0.05:
        near = min(1.0, base + 0.05)
    return Levels(base, near)


def _agglomerate(X: np.ndarray, sim: float, n_clusters: int | None = None) -> np.ndarray:
    from sklearn.cluster import AgglomerativeClustering

    n = len(X)
    if n == 1:
        return np.zeros(1, dtype=int)
    if n_clusters is not None:
        n_clusters = max(1, min(n_clusters, n))
        if n_clusters == 1:
            return np.zeros(n, dtype=int)
        model = AgglomerativeClustering(n_clusters=n_clusters, metric="cosine", linkage="average")
    else:
        model = AgglomerativeClustering(
            n_clusters=None, metric="cosine", linkage="average", distance_threshold=max(1e-6, 1.0 - sim))
    return model.fit_predict(X)


def _cluster_direct(X: np.ndarray, sim: float) -> np.ndarray:
    n = len(X)
    if n <= DIRECT_LIMIT:
        return _agglomerate(X, sim)
    # Large collections: compress into micro-clusters first (memory stays reasonable), then link those.
    from sklearn.cluster import MiniBatchKMeans

    k = min(2500, max(50, n // 6))
    km = MiniBatchKMeans(n_clusters=k, random_state=0, batch_size=4096, n_init=3).fit(X)
    C = km.cluster_centers_
    C = C / np.maximum(np.linalg.norm(C, axis=1, keepdims=True), 1e-12)
    return _agglomerate(C, sim)[km.labels_]


def _scores(X: np.ndarray, centroids: np.ndarray, levels: Levels, sim: float) -> np.ndarray:
    """Average similarity to each group, rescaled so 1.0 == the linkage threshold, 0.0 == unrelated."""
    raw = X @ centroids.T
    return (raw - levels.base) / max(sim - levels.base, 1e-6)


def cluster(X: np.ndarray, detail: str = "normal", levels: Levels | None = None
            ) -> tuple[list[list[int]], dict[int, float], Levels, float]:
    """Groups vectors. Returns (groups of indices, score per assigned index, levels, linkage similarity)."""
    alpha, min_size, max_size = DETAIL.get(detail, DETAIL["normal"])
    n = len(X)
    levels = levels or measure_levels(X)
    sim = levels.at(alpha)
    if n < 2:
        return [], {}, levels, sim
    if n < 40:
        min_size = 2
    labels = _cluster_direct(X, sim)

    # Split groups that are too big to browse comfortably, by re-clustering them more strictly.
    groups: list[list[int]] = []
    pending = [(np.flatnonzero(labels == lab).tolist(), 0) for lab in np.unique(labels)]
    while pending:
        g, depth = pending.pop()
        if len(g) > max_size and depth < 3:
            # split into a few coherent parts (k-means follows the main sub-structure of the group)
            from sklearn.cluster import KMeans

            k = math.ceil(len(g) / (0.75 * max_size))
            sub = KMeans(n_clusters=k, n_init=3, random_state=0).fit_predict(X[g])
            parts = [[g[i] for i in np.flatnonzero(sub == s)] for s in np.unique(sub)]
            if len(parts) > 1:
                pending.extend((p, depth + 1) for p in parts)
                continue
        groups.append(g)

    keep = [g for g in groups if len(g) >= min_size]
    orphans = [i for g in groups if len(g) < min_size for i in g]
    if not keep:
        # Everything is unique: try once more, more leniently.
        labels = _cluster_direct(X, levels.at(alpha * 0.6))
        keep = [g for lab in np.unique(labels) if len(g := np.flatnonzero(labels == lab).tolist()) >= min_size]
        kept = {i for g in keep for i in g}
        orphans = [i for i in range(n) if i not in kept]
        if not keep:
            return [], {}, levels, sim

    cents = np.stack([X[g].mean(0) for g in keep])
    score: dict[int, float] = {}
    for g, c in zip(keep, cents):
        s = _scores(X[g], c[None, :], levels, sim)[:, 0]
        for i, v in zip(g, s):
            score[i] = float(min(1.0, max(0.0, v)))

    # Second chance for files that didn't form a group: join the most similar group if close enough.
    leftover = []
    if orphans:
        S = _scores(X[orphans], cents, levels, sim)
        best = S.argmax(1)
        for i, b, row in zip(orphans, best, S):
            if row[b] >= LOOSE:
                keep[b].append(i)
                score[i] = float(min(1.0, row[b]))
            else:
                leftover.append(i)
    # Files that are still alone may form small groups among themselves.
    if len(leftover) >= max(4, min_size * 2):
        sub = _agglomerate(X[leftover], sim)
        for s in np.unique(sub):
            g = [leftover[i] for i in np.flatnonzero(sub == s)]
            if len(g) >= min_size:
                keep.append(g)
                c = X[g].mean(0)
                for i, v in zip(g, _scores(X[g], c[None, :], levels, sim)[:, 0]):
                    score[i] = float(min(1.0, max(0.0, v)))
    return keep, score, levels, sim


# ---------------------------------------------------------------------------------------------------------------
# Naming
# ---------------------------------------------------------------------------------------------------------------
class Terms:
    """Which keywords/phrases each document contains, as a compact sparse 0/1 matrix (documents x terms).
    Scales to tens of thousands of documents without holding Python sets per document."""

    def __init__(self, M, vocab):
        self.M = M.tocsr()
        self.vocab = np.asarray(vocab, dtype=object)

    @classmethod
    def build(cls, texts: list[str], max_words: int = 600) -> "Terms":
        from sklearn.feature_extraction.text import CountVectorizer

        if not texts:
            from scipy.sparse import csr_matrix

            return cls(csr_matrix((0, 0), dtype=np.uint8), [])
        cv = CountVectorizer(analyzer=lambda t: list(terms_and_phrases(t, max_words)), binary=True,
                             dtype=np.uint8, min_df=2 if len(texts) >= 200 else 1)
        try:
            M = cv.fit_transform(texts)
            vocab = cv.get_feature_names_out()
        except ValueError:  # nothing meaningful at all
            from scipy.sparse import csr_matrix

            return cls(csr_matrix((len(texts), 0), dtype=np.uint8), [])
        return cls(M, vocab)

    def __len__(self) -> int:
        return self.M.shape[0]

    def df(self, rows=None) -> np.ndarray:
        M = self.M if rows is None else self.M[rows]
        return np.asarray(M.sum(axis=0, dtype=np.int64)).ravel()


def keyword_table(terms: Terms, groups: list[list[int]], top: int = 10) -> list[list[str]]:
    """Class-based TF-IDF: words that are frequent inside a group and rare elsewhere."""
    N = max(1, len(terms))
    if terms.M.shape[1] == 0:
        return [[] for _ in groups]
    idf = np.log(1 + N / np.maximum(terms.df(), 1))
    phrase = np.array([" " in t for t in terms.vocab])
    out = []
    for g in groups:
        n_g = len(g)
        df_g = terms.df(g)
        min_df = 1 if n_g <= 3 else 2
        score = (df_g / n_g) * idf * np.where(phrase, 1.12, 1.0)
        score[df_g < min_df] = 0
        order = np.argsort(-score)[: top * 8]
        chosen: list[str] = []
        for k in order:
            if score[k] <= 0:
                break
            t = terms.vocab[k]
            tw = set(t.split())
            if any(t in c or c in t or (len(tw & set(c.split())) and " " in c and " " in t) for c in chosen):
                continue
            chosen.append(t)
            if len(chosen) >= top:
                break
        out.append(chosen)
    return out


def name_from_keywords(kw: list[str], fallback: str, sample: str = "") -> str:
    """'plafond de dépenses' + 'indemnités kilométriques' -> 'Plafond de dépenses & Indemnités kilométriques'."""
    if not kw:
        return fallback
    first = kw[0]
    second = next((k for k in kw[1:] if not set(k.split()) & set(first.split())), None)

    def fmt(k):
        return sentence_case(surface_form(k, sample)) if sample else nice_case(k)

    if second and len(first) + len(second) <= 46:
        return f"{fmt(first)} & {fmt(second)}"
    return fmt(first)


def presence(terms: Terms, members: list[int], share: float = 0.4) -> set[str]:
    """Terms found in at least `share` of the given documents."""
    if terms.M.shape[1] == 0 or not members:
        return set()
    need = max(1, math.ceil(share * len(members)))
    return set(terms.vocab[terms.df(members) >= need])


def area_keyword(area_topics: list[int], topic_presence: list[set[str]], topic_keywords: list[list[str]]
                 ) -> str | None:
    """A word shared by most topics of an area and rare in the other areas' topics."""
    T = len(topic_presence)
    df = Counter()
    for p in topic_presence:
        df.update(p)
    if len(area_topics) == 1:
        # a single subject: its main keyword/phrase is the clearest name ("Vehicle routing")
        return (topic_keywords[area_topics[0]] or [None])[0]
    best, best_score = None, 0.0
    # candidates must be real subject words of the sub-folders (not letter boilerplate shared by everything)
    subject_words = {k for ti in area_topics for k in topic_keywords[ti][:8]}
    for t in set().union(*(topic_presence[ti] for ti in area_topics)) & subject_words:
        cov = sum(t in topic_presence[ti] for ti in area_topics) / len(area_topics)
        if cov < 0.5:
            continue
        score = cov * math.log(1 + T / df[t]) * (1.1 if " " in t else 1.0)
        if score > best_score:
            best, best_score = t, score
    # only a word that nearly all of the area's topics share is a trustworthy name for the area
    if best and best_score >= 0.67 * math.log(1 + T / len(area_topics)):
        return best
    return None


def _date(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d")


def _dominant_folder(parents: list[str], share: float = 0.5) -> str | None:
    names = [p for p in parents if p and p.lower().strip() not in GENERIC_FOLDERS and not re.fullmatch(r"[\d\W_]+", p)]
    if not names:
        return None
    name, c = Counter(names).most_common(1)[0]
    return name if c >= share * len(parents) else None


# ---------------------------------------------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------------------------------------------
AREA_FACTOR = 0.35   # areas link topics at a looser similarity than the one that defines a topic


def area_threshold(levels: Levels, detail: str) -> float:
    return levels.at(DETAIL.get(detail, DETAIL["normal"])[0] * AREA_FACTOR)


def group_into_areas(means: np.ndarray, levels: Levels, detail: str = "normal") -> np.ndarray:
    """Links topics into broad areas. `means` are the topics' (unnormalised) mean vectors, so
    means[a] @ means[b] is the average similarity between their documents - on the same scale as `levels`.
    Unrelated topics are never forced together; only very many areas are capped."""
    from sklearn.cluster import AgglomerativeClustering

    T = len(means)
    if T < 2:
        return np.zeros(T, dtype=int)
    S = means @ means.T
    D = np.clip(1.0 - S, 0.0, None)
    np.fill_diagonal(D, 0.0)
    D = (D + D.T) / 2
    labels = AgglomerativeClustering(n_clusters=None, metric="precomputed", linkage="average",
                                     distance_threshold=max(1e-6, 1.0 - area_threshold(levels, detail))
                                     ).fit_predict(D)
    max_areas = max(8, min(25, round(2 * math.sqrt(T))))
    if len(set(labels)) > max_areas:
        labels = AgglomerativeClustering(n_clusters=max_areas, metric="precomputed", linkage="average"
                                         ).fit_predict(D)
    return labels


def organize_documents(X: np.ndarray, doc_terms: Terms, titles: list[str], detail: str = "normal",
                       namer=None, levels: Levels | None = None, texts: list[str] | None = None
                       ) -> tuple[list[Topic], Levels, float]:
    """Groups documents by subject and names the groups. `texts` (optional) are used to keep original
    spelling in names (acronyms, accents, apostrophes)."""
    groups, score, levels, sim = cluster(X, detail, levels)
    topics = [Topic(members=g, centroid=X[g].mean(0), scores={i: score.get(i, 0.0) for i in g}) for g in groups]
    if not topics:
        return [], levels, sim

    # Two levels: topics whose documents are, on average, still clearly related form a broad area.
    area_labels = group_into_areas(np.stack([t.centroid for t in topics]), levels, detail)

    for t, kw in zip(topics, keyword_table(doc_terms, [t.members for t in topics])):
        t.keywords = kw
    topic_presence = [presence(doc_terms, t.members) for t in topics]
    texts = texts or titles

    def sample_of(members: list[int]) -> str:
        return " ".join(texts[i][:3000] for i in members[:40])

    area_groups = defaultdict(list)
    for ti, a in enumerate(area_labels):
        area_groups[int(a)].append(ti)

    used_areas: set[str] = set()
    for tis in sorted(area_groups.values(), key=lambda v: -sum(len(topics[i].members) for i in v)):
        names_in_area: set[str] = set()
        for ti in tis:
            t = topics[ti]
            name = None
            if namer is not None:
                examples = [titles[i] for i in sorted(t.members, key=lambda i: -t.scores.get(i, 0))[:10]]
                name = namer.name_topic(t.keywords, examples)
            name = safe_name(name or name_from_keywords(t.keywords, "Miscellaneous", sample_of(t.members)))
            t.name = unique_name(name, names_in_area)
            names_in_area.add(t.name)
        aname = None
        members = [i for ti in tis for i in topics[ti].members]
        if namer is not None:
            kw = [k for ti in tis for k in topics[ti].keywords[:3]]
            children = [topics[ti].name for ti in tis]
            taken = {c.lower() for c in children} | {a.lower() for a in used_areas}
            aname = namer.name_area(children, kw)
            if aname and aname.lower() in taken:   # must be broader than its sub-folders: ask once more
                aname = namer.name_area(children, kw, avoid=sorted(taken))
                if aname and aname.lower() in taken:
                    aname = None
        if not aname:
            k = area_keyword(tis, topic_presence, [t.keywords for t in topics])
            if k:
                aname = sentence_case(surface_form(k, sample_of(members)))
            else:  # no shared word: name the area after its biggest subjects
                big = sorted(tis, key=lambda ti: -len(topics[ti].members))[:3]
                firsts = [sentence_case(surface_form(topics[ti].keywords[0], sample_of(topics[ti].members)))
                          for ti in big if topics[ti].keywords]
                aname = (", ".join(firsts[:-1]) + " & " + firsts[-1]) if len(firsts) > 1 else \
                    (firsts[0] if firsts else "Documents")
        aname = unique_name(safe_name(aname, max_len=64), used_areas)
        used_areas.add(aname)
        for ti in tis:
            topics[ti].area = aname
    return topics, levels, sim


# ---------------------------------------------------------------------------------------------------------------
# Pictures and videos
# ---------------------------------------------------------------------------------------------------------------
@dataclass
class VisualItem:
    kind: str                    # image | video | design
    taken: float | None          # capture time (EXIF / video metadata), else None
    mtime: float
    parent: str                  # name of the folder the file was in
    screen: bool                 # looks like a screenshot / screen recording
    camera: bool                 # has camera EXIF
    label: int | None = None     # best matching LABELS index (image model) or None
    label_score: float = 0.0
    area_scores: dict[str, float] | None = None


def visual_area(it: VisualItem) -> str:
    if it.area_scores:
        area = max(it.area_scores, key=it.area_scores.get)
    elif it.screen:
        area = SCREENSHOTS
    elif it.camera or (it.taken and it.kind in ("image", "video")):
        area = PHOTOS
    elif it.kind == "design":
        area = GRAPHICS
    else:
        area = OTHER_IMAGES
    if it.kind == "video":
        if it.screen or area == SCREENSHOTS:
            return RECORDINGS
        if area == PHOTOS:
            return PHOTOS
        return VIDEOS
    if it.camera and area == SCREENSHOTS and not it.area_scores:
        return PHOTOS
    return area


def segment_events(times: list[float], gap_hours: float = 8.0) -> list[list[int]]:
    order = sorted(range(len(times)), key=lambda i: times[i])
    segs: list[list[int]] = []
    for i in order:
        if segs and times[i] - times[segs[-1][-1]] <= gap_hours * 3600:
            segs[-1].append(i)
        else:
            segs.append([i])
    return segs


def event_name(items: list[VisualItem], members: list[int]) -> str:
    ts = [items[i].taken for i in members if items[i].taken]
    start, end = min(ts), max(ts)
    days = (datetime.fromtimestamp(end).date() - datetime.fromtimestamp(start).date()).days + 1
    label = _dominant_folder([items[i].parent for i in members])
    if not label:
        labs = [LABELS[items[i].label][1] for i in members
                if items[i].label is not None and LABELS[items[i].label][2] == PHOTOS]
        generic = {"Portraits", "Selfies", "Group photos", "Headshots"}
        specific = [lab for lab in labs if lab not in generic]
        pool = specific if len(specific) >= 0.3 * max(1, len(labs)) else labs
        label = Counter(pool).most_common(1)[0][0] if pool else "Photos"
    suffix = f" ({days} days)" if days > 1 else ""
    return safe_name(f"{_date(start)} {label}{suffix}", max_len=70)


def organize_visual(items: list[VisualItem], X: np.ndarray, detail: str = "normal") -> list[Topic]:
    """Groups pictures/videos. Photos of events are grouped by date; everything else by what it shows."""
    _, min_size, max_size = DETAIL.get(detail, DETAIL["normal"])
    by_area: dict[str, list[int]] = defaultdict(list)
    for i, it in enumerate(items):
        by_area[visual_area(it)].append(i)

    topics: list[Topic] = []
    for area, idx in by_area.items():
        if area == PHOTOS:
            dated = [i for i in idx if items[i].taken]
            undated = [i for i in idx if not items[i].taken]
            segs = [[dated[j] for j in s] for s in segment_events([items[i].taken for i in dated])]
            small: dict[str, list[int]] = defaultdict(list)
            for s in segs:
                if len(s) >= 3:
                    t0 = min(items[i].taken for i in s)
                    t1 = max(items[i].taken for i in s)
                    topics.append(Topic(members=s, kind="event", area=area, name=event_name(items, s),
                                        t_start=t0, t_end=t1, centroid=X[s].mean(0) if len(X) else None))
                else:
                    for i in s:
                        small[datetime.fromtimestamp(items[i].taken).strftime("%Y-%m")].append(i)
            for month, s in sorted(small.items()):
                topics.append(Topic(members=s, kind="month", area=area, name=f"{month} Various photos",
                                    label=month, centroid=X[s].mean(0) if len(X) else None))
            if undated:
                topics.extend(_group_by_label_or_folder(items, X, undated, area, "Undated photos", max_size))
        else:
            fallback = {SCREENSHOTS: "Screenshots", RECORDINGS: "Screen recordings", VIDEOS: "Videos",
                        SCANS: "Scans", GRAPHICS: "Images", OTHER_IMAGES: "Images"}.get(area, "Images")
            topics.extend(_group_by_label_or_folder(items, X, idx, area, fallback, max_size))
    # unique names per area
    seen: dict[str, set[str]] = defaultdict(set)
    for t in topics:
        t.name = unique_name(safe_name(t.name, max_len=70), seen[t.area])
        seen[t.area].add(t.name)
        t.scores = {i: 1.0 for i in t.members}
    return topics


def _group_by_label_or_folder(items, X, idx, area, fallback, max_size) -> list[Topic]:
    groups: dict[str, list[int]] = defaultdict(list)
    for i in idx:
        it = items[i]
        if it.label is not None and LABELS[it.label][2] == area:
            key = LABELS[it.label][1]
        elif it.label is not None and area in (RECORDINGS, VIDEOS):
            key = LABELS[it.label][1]
        else:
            folder = it.parent if it.parent and it.parent.lower() not in GENERIC_FOLDERS else None
            key = folder or datetime.fromtimestamp(it.taken or it.mtime).strftime("%Y-%m")
        groups[key].append(i)
    out = []
    for key, members in groups.items():
        if len(members) > max_size:
            # very large groups are split by year so folders stay browsable
            by_year: dict[str, list[int]] = defaultdict(list)
            for i in members:
                by_year[datetime.fromtimestamp(items[i].taken or items[i].mtime).strftime("%Y")].append(i)
            for y, m in sorted(by_year.items()):
                out.append(Topic(members=m, kind="visual", area=area, name=f"{key} {y}", label=key,
                                 centroid=X[m].mean(0) if len(X) else None))
        else:
            out.append(Topic(members=members, kind="visual", area=area, name=key or fallback, label=key,
                             centroid=X[members].mean(0) if len(X) else None))
    return out
