"""Prints where every file went (folder <- file, and why), from a destination's database.

    python tools/show_result.py <destination folder>
"""

import sqlite3
import sys
from collections import defaultdict
from pathlib import Path


def main(dest: Path) -> None:
    db = sqlite3.connect(str(dest / "_FileSorter" / "state.sqlite"))
    rows = db.execute("""SELECT p.dest, f.name, p.role, p.reason, f.label, f.area_hint, p.confidence
                         FROM placements p JOIN files f ON f.id = p.file_id ORDER BY p.dest""").fetchall()
    by_folder = defaultdict(list)
    for dest_rel, name, role, reason, label, area, conf in rows:
        folder = str(Path(dest_rel).parent) if dest_rel else "(not copied)"
        by_folder[folder].append((name, role, f"{reason or ''} ({conf or 0:.2f})", label, area))
    for folder in sorted(by_folder):
        print(f"\n{folder}/")
        for name, role, reason, label, area in by_folder[folder]:
            extra = f" [image: {area}]" if area else ""
            print(f"    {name:<45} {role:<9} {reason or ''}{extra}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
