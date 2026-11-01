"""Adds new, changed and duplicate files to a sample folder (used by the incremental-run test).

    python tests/add_new_files.py <sample folder>
"""

import random
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import make_sample as ms  # noqa: E402


def main(root: Path) -> None:
    rng = random.Random(9)
    new = root / "Downloads" / "new"
    new.mkdir(exist_ok=True)
    # one more document about an existing subject
    ms.make_docx(new / "Note de frais juillet.docx", "Note de frais juillet", ms.paragraph("expenses_fr", rng, 8))
    # a brand-new subject
    ms.TOPICS["csr"] = ("en", [
        "sustainability report", "carbon footprint", "CSR strategy", "greenhouse gas emissions", "renewable energy",
        "waste reduction", "supplier code of conduct", "biodiversity", "ESG rating", "scope 3 emissions",
        "energy efficiency", "recycling programme", "climate targets", "social impact", "diversity and inclusion"])
    for n in ("Sustainability report 2024", "Carbon footprint assessment", "ESG rating feedback",
              "Climate targets roadmap", "Recycling programme update"):
        ms.make_docx(new / f"{n}.docx", n, ms.paragraph("csr", rng, 8))
    # photos of a new event, and one more photo of an existing event
    for k in range(4):
        ms.make_photo(new / f"IMG_9{k}.jpg", f"2025:03:07 1{k}:00:00", seed=200 + k)
    ms.make_photo(new / "IMG_extra_annecy.jpg", "2024:06:14 18:40:00", seed=300)
    # a renamed duplicate
    shutil.copy(next(root.rglob("Password policy.pdf")), new / "policy_final.pdf")
    # a changed file
    p = next(root.rglob("Quiz Excel.txt"))
    time.sleep(1)
    p.write_text(p.read_text(encoding="utf-8") + "\nQuestion bonus: tableau croisé dynamique.", encoding="utf-8")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
