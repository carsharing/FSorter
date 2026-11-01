"""Local SQLite database kept inside the destination folder (<destination>/_FileSorter/state.sqlite).

It remembers what was analysed (so later runs only look at new or changed files), the folders the app created,
and where every file was copied.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

import numpy as np

ANALYSIS_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS files(
    id INTEGER PRIMARY KEY,
    path TEXT NOT NULL UNIQUE,
    root TEXT, rel TEXT, name TEXT, ext TEXT, kind TEXT,
    size INTEGER, mtime REAL,
    analyzed INTEGER DEFAULT 0,
    error TEXT,
    title TEXT, embed_text TEXT, meta TEXT,
    route TEXT,
    head_hash TEXT, full_hash TEXT, text_fp TEXT,
    phash INTEGER, dhash INTEGER, detail REAL, aspect REAL, csig BLOB,
    vhashes TEXT, duration REAL, taken REAL,
    label INTEGER, label_score REAL, area_hint TEXT,
    temb BLOB, tspace TEXT, vemb BLOB, vspace TEXT,
    seen_run INTEGER
);
CREATE INDEX IF NOT EXISTS files_size ON files(size);
CREATE INDEX IF NOT EXISTS files_fp ON files(text_fp);
CREATE INDEX IF NOT EXISTS files_fullhash ON files(full_hash);

CREATE TABLE IF NOT EXISTS categories(
    id INTEGER PRIMARY KEY,
    area TEXT, name TEXT, folder TEXT UNIQUE,
    kind TEXT,
    space TEXT, centroid BLOB, n INTEGER,
    keywords TEXT, label TEXT, t_start REAL, t_end REAL,
    created REAL
);

CREATE TABLE IF NOT EXISTS placements(
    file_id INTEGER PRIMARY KEY REFERENCES files(id) ON DELETE CASCADE,
    category_id INTEGER,
    role TEXT,
    dup_of INTEGER,
    reason TEXT,
    confidence REAL,
    dest TEXT,
    status TEXT DEFAULT 'pending',
    copied_size INTEGER, copied_mtime REAL, copied_at REAL, run_id INTEGER
);
CREATE INDEX IF NOT EXISTS placements_cat ON placements(category_id);

CREATE TABLE IF NOT EXISTS runs(id INTEGER PRIMARY KEY, started REAL, finished REAL, kind TEXT, stats TEXT);
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
"""


def to_blob(v: np.ndarray | None) -> bytes | None:
    return None if v is None else np.asarray(v, dtype=np.float32).tobytes()


def from_blob(b: bytes | None) -> np.ndarray | None:
    return None if b is None else np.frombuffer(b, dtype=np.float32)


class State:
    def __init__(self, folder: Path):
        folder.mkdir(parents=True, exist_ok=True)
        self.folder = folder
        self.db = sqlite3.connect(str(folder / "state.sqlite"), timeout=30, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript(SCHEMA)
        self.db.commit()

    def close(self):
        try:
            self.db.commit()
            self.db.close()
        except sqlite3.Error:
            pass

    # -- meta ---------------------------------------------------------------------------------------------------
    def get_meta(self, key: str, default=None):
        r = self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return json.loads(r[0]) if r else default

    def set_meta(self, key: str, value) -> None:
        self.db.execute("INSERT OR REPLACE INTO meta(key, value) VALUES(?, ?)", (key, json.dumps(value)))

    # -- runs ---------------------------------------------------------------------------------------------------
    def start_run(self, kind: str) -> int:
        cur = self.db.execute("INSERT INTO runs(started, kind) VALUES(?, ?)", (time.time(), kind))
        self.db.commit()
        return cur.lastrowid

    def finish_run(self, run_id: int, stats: dict) -> None:
        self.db.execute("UPDATE runs SET finished=?, stats=? WHERE id=?", (time.time(), json.dumps(stats), run_id))
        self.db.commit()

    # -- files --------------------------------------------------------------------------------------------------
    def known_files(self) -> dict[str, sqlite3.Row]:
        return {r["path"]: r for r in self.db.execute(
            "SELECT id, path, size, mtime, analyzed FROM files")}

    def upsert_file(self, rec: dict) -> int:
        cols = list(rec)
        row = self.db.execute("SELECT id FROM files WHERE path=?", (rec["path"],)).fetchone()
        if row:
            sets = ", ".join(f"{c}=?" for c in cols if c != "path")
            self.db.execute(f"UPDATE files SET {sets} WHERE id=?", [rec[c] for c in cols if c != "path"] + [row[0]])
            return row[0]
        cur = self.db.execute(
            f"INSERT INTO files({', '.join(cols)}) VALUES({', '.join('?' * len(cols))})", [rec[c] for c in cols])
        return cur.lastrowid

    def update_file(self, file_id: int, **fields) -> None:
        if fields:
            sets = ", ".join(f"{k}=?" for k in fields)
            self.db.execute(f"UPDATE files SET {sets} WHERE id=?", [*fields.values(), file_id])

    def files(self, where: str = "1=1", params=()) -> list[sqlite3.Row]:
        return self.db.execute(f"SELECT * FROM files WHERE {where}", params).fetchall()

    # -- categories ---------------------------------------------------------------------------------------------
    def categories(self) -> list[sqlite3.Row]:
        return self.db.execute("SELECT * FROM categories ORDER BY folder").fetchall()

    def add_category(self, **c) -> int:
        c.setdefault("created", time.time())
        cols = list(c)
        cur = self.db.execute(
            f"INSERT INTO categories({', '.join(cols)}) VALUES({', '.join('?' * len(cols))})", list(c.values()))
        return cur.lastrowid

    def update_category(self, cat_id: int, **fields) -> None:
        if fields:
            sets = ", ".join(f"{k}=?" for k in fields)
            self.db.execute(f"UPDATE categories SET {sets} WHERE id=?", [*fields.values(), cat_id])

    # -- placements ---------------------------------------------------------------------------------------------
    def placement(self, file_id: int) -> sqlite3.Row | None:
        return self.db.execute("SELECT * FROM placements WHERE file_id=?", (file_id,)).fetchone()

    def set_placement(self, file_id: int, **p) -> None:
        existing = self.placement(file_id)
        if existing:
            sets = ", ".join(f"{k}=?" for k in p)
            self.db.execute(f"UPDATE placements SET {sets} WHERE file_id=?", [*p.values(), file_id])
        else:
            p["file_id"] = file_id
            cols = list(p)
            self.db.execute(
                f"INSERT INTO placements({', '.join(cols)}) VALUES({', '.join('?' * len(cols))})", list(p.values()))

    def placements(self) -> list[sqlite3.Row]:
        return self.db.execute("SELECT * FROM placements").fetchall()

    def taken_dests(self) -> set[str]:
        return {r[0].lower() for r in self.db.execute("SELECT dest FROM placements WHERE dest IS NOT NULL")}

    def commit(self):
        self.db.commit()
