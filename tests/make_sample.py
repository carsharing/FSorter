"""Builds a realistic, mixed 'office' folder for tests and demos (French + English).

    python tests/make_sample.py <folder>
"""

from __future__ import annotations

import random
import shutil
import sys
from email.message import EmailMessage
from pathlib import Path

TOPICS = {
    "expenses_fr": ("fr", [
        "note de frais", "remboursement", "justificatif", "déplacement professionnel", "indemnités kilométriques",
        "ticket de restaurant", "hôtel", "billet de train", "validation du manager", "service comptabilité",
        "plafond de dépenses", "carte affaires", "TVA récupérable", "avance de frais", "mission client à Lyon",
    ]),
    "recruiting": ("en", [
        "job description", "candidate interview", "hiring manager", "onboarding checklist", "offer letter",
        "probation period", "recruitment pipeline", "screening call", "salary range", "background check",
        "first day schedule", "welcome pack", "employee handbook", "buddy program", "HR business partner",
    ]),
    "excel_training_fr": ("fr", [
        "formation Excel", "tableau croisé dynamique", "fonction RECHERCHEV", "mise en forme conditionnelle",
        "graphiques", "filtres avancés", "macros VBA", "exercice pratique", "module débutant", "raccourcis clavier",
        "validation des données", "fonction SOMME.SI", "consolidation des feuilles", "support de cours", "stagiaires",
    ]),
    "brand": ("en", [
        "brand guidelines", "logo redesign", "colour palette", "typography", "visual identity", "marketing campaign",
        "social media assets", "tone of voice", "brand awareness", "creative agency", "mood board", "tagline",
        "launch event", "product positioning", "target audience",
    ]),
    "security": ("en", [
        "phishing awareness", "password policy", "multi-factor authentication", "security incident", "malware",
        "suspicious email", "VPN access", "data breach", "IT helpdesk", "endpoint protection", "patch Tuesday",
        "access rights review", "security training", "encryption", "firewall rules",
    ]),
    "budget_fr": ("fr", [
        "budget prévisionnel", "exercice 2025", "charges de personnel", "investissements", "trésorerie",
        "chiffre d'affaires", "marge brute", "contrôle de gestion", "écart budgétaire", "prévisions trimestrielles",
        "dotations aux amortissements", "comité de direction", "arbitrage", "reporting mensuel", "centre de coûts",
    ]),
    "suppliers_fr": ("fr", [
        "contrat fournisseur", "conditions générales d'achat", "bon de commande", "appel d'offres", "délai de livraison",
        "pénalités de retard", "avenant au contrat", "résiliation", "clause de confidentialité", "service achats",
        "référencement fournisseur", "négociation tarifaire", "facture fournisseur", "litige", "garantie",
    ]),
    "routing": ("en", [
        "vehicle routing problem", "car sharing", "instance generator", "time windows", "depot", "heuristic",
        "travel time matrix", "optimisation", "fleet size", "relocation of vehicles", "demand forecast",
        "simulation results", "benchmark instances", "column generation", "station capacity",
    ]),
}

FILLER = {
    "fr": ["Merci de bien vouloir", "Comme convenu lors de notre réunion,", "Veuillez trouver ci-joint",
           "Pour rappel,", "Il est important de noter que", "Nous avons constaté que", "Suite à notre échange,",
           "Le document précise que", "En conclusion,", "Concernant le point suivant,"],
    "en": ["Please find attached", "As discussed in our meeting,", "Just a reminder that", "We noticed that",
           "Following our call,", "The document explains that", "In summary,", "Regarding the next step,",
           "It is important to note that", "Could you please review"],
}


def paragraph(topic: str, rng: random.Random, n: int = 6) -> str:
    lang, terms = TOPICS[topic]
    out = []
    for _ in range(n):
        t = rng.sample(terms, 3)
        out.append(f"{rng.choice(FILLER[lang])} {t[0]}, {t[1]} et {t[2]}." if lang == "fr"
                   else f"{rng.choice(FILLER[lang])} {t[0]}, {t[1]} and {t[2]}.")
    return " ".join(out)


def make_docx(path: Path, title: str, body: str):
    import docx

    d = docx.Document()
    d.core_properties.title = title
    d.add_heading(title, 1)
    for chunk in body.split(". "):
        d.add_paragraph(chunk)
    d.save(path)


def make_xlsx(path: Path, title: str, topic: str, rng: random.Random):
    import openpyxl

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Synthèse" if TOPICS[topic][0] == "fr" else "Summary"
    ws.append([title])
    for term in rng.sample(TOPICS[topic][1], 8):
        ws.append([term, rng.randint(100, 9000), rng.random()])
    wb.properties.title = title
    wb.save(path)


def make_pptx(path: Path, title: str, topic: str, rng: random.Random):
    from pptx import Presentation

    prs = Presentation()
    s = prs.slides.add_slide(prs.slide_layouts[0])
    s.shapes.title.text = title
    s.placeholders[1].text = paragraph(topic, rng, 1)
    for _ in range(3):
        s = prs.slides.add_slide(prs.slide_layouts[1])
        s.shapes.title.text = rng.choice(TOPICS[topic][1]).capitalize()
        s.placeholders[1].text = paragraph(topic, rng, 2)
    prs.save(path)


def make_pdf(path: Path, title: str, body: str):
    from fpdf import FPDF

    pdf = FPDF()
    pdf.set_title(title)
    pdf.add_page()
    pdf.set_font("Helvetica", size=16)
    pdf.multi_cell(0, 10, title.encode("latin-1", "replace").decode("latin-1"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", size=11)
    pdf.multi_cell(0, 6, body.encode("latin-1", "replace").decode("latin-1"), new_x="LMARGIN", new_y="NEXT")
    pdf.output(str(path))


def make_eml(path: Path, subject: str, body: str, attach: str | None = None):
    m = EmailMessage()
    m["From"] = "Camille Martin <camille.martin@example.com>"
    m["To"] = "equipe@example.com"
    m["Subject"] = subject
    m["Date"] = "Tue, 04 Jun 2024 09:30:00 +0200"
    m.set_content(body)
    if attach:
        m.add_attachment(b"%PDF-1.4 fake", maintype="application", subtype="pdf", filename=attach)
    path.write_bytes(bytes(m))


def make_photo(path: Path, when: str, seed: int, people: bool = True, size=(1600, 1200)):
    from PIL import Image, ImageDraw

    r = random.Random(seed)
    img = Image.new("RGB", size, (r.randint(60, 200), r.randint(120, 220), r.randint(60, 200)))
    d = ImageDraw.Draw(img)
    for _ in range(25):
        x, y = r.randint(0, size[0]), r.randint(0, size[1])
        rad = r.randint(30, 160)
        d.ellipse([x - rad, y - rad, x + rad, y + rad], fill=(r.randint(0, 255), r.randint(0, 255), r.randint(0, 255)))
    if people:
        for k in range(5):
            x = 150 + k * 280
            d.ellipse([x, 400, x + 120, 520], fill=(230, 190, 160))
            d.rectangle([x - 20, 520, x + 140, 900], fill=(r.randint(0, 80), r.randint(0, 80), r.randint(100, 200)))
    exif = Image.Exif()
    exif[271] = "Apple"
    exif[272] = "iPhone 13"
    exif[306] = when
    img.save(path, quality=88, exif=exif.tobytes())


def make_screenshot(path: Path, lines: list[str], seed: int):
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (1920, 1080), "white")
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 1920, 60], fill=(30, 60, 140))
    try:
        font = ImageFont.load_default(size=34)
    except TypeError:  # old Pillow
        font = ImageFont.load_default()
    for i, line in enumerate(lines):
        d.text((60, 120 + i * 60), line, fill="black", font=font)
    img.save(path)


def make_logo(path: Path, seed: int, color):
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (800, 800), "white")
    d = ImageDraw.Draw(img)
    d.ellipse([200, 200, 600, 600], fill=color)
    d.rectangle([360, 120 + seed * 10, 440, 680], fill="white")
    img.save(path)


def make_video(path: Path, frames: int = 48, text: str = "Budget review"):
    import av
    import numpy as np
    from PIL import Image, ImageDraw

    with av.open(str(path), "w") as c:
        s = c.add_stream("mpeg4", rate=12)
        s.width, s.height, s.pix_fmt = 1280, 720, "yuv420p"
        for i in range(frames):
            img = Image.new("RGB", (1280, 720), "white")
            d = ImageDraw.Draw(img)
            d.rectangle([0, 0, 1280, 50], fill=(50, 50, 50))
            d.text((40, 100 + i), f"{text} - slide {i // 12 + 1}", fill="black")
            frame = av.VideoFrame.from_ndarray(np.asarray(img), format="rgb24")
            for pkt in s.encode(frame):
                c.mux(pkt)
        for pkt in s.encode():
            c.mux(pkt)


def build(root: Path, seed: int = 1) -> Path:
    rng = random.Random(seed)
    if root.exists():
        shutil.rmtree(root)
    docs = root / "Documents"
    dl = root / "Downloads"
    desk = root / "Desktop"
    pics = root / "Pictures"
    for d in (docs, dl, desk, pics):
        d.mkdir(parents=True)

    titles = {
        "expenses_fr": ["Note de frais juin", "Remboursement mission Lyon", "Procédure notes de frais",
                        "Justificatifs déplacement", "Frais kilométriques mai", "Politique voyages"],
        "recruiting": ["Onboarding checklist", "Job description data analyst", "Interview guide",
                       "Offer letter template", "New hire first week", "Recruitment pipeline Q3"],
        "excel_training_fr": ["Formation Excel module 1", "Exercices tableaux croisés", "Support cours Excel avancé",
                              "Macros VBA initiation", "Formation Excel module 2", "Quiz Excel"],
        "brand": ["Brand guidelines v2", "Logo redesign brief", "Campaign plan spring", "Social media calendar",
                  "Agency proposal", "Launch event plan"],
        "security": ["Phishing awareness", "Password policy", "Security incident report", "VPN setup guide",
                     "MFA rollout", "Access review"],
        "budget_fr": ["Budget prévisionnel 2025", "Reporting mensuel mars", "Prévisions T2", "Analyse écarts",
                      "Plan investissements", "Comité direction budget"],
        "suppliers_fr": ["Contrat fournisseur Acme", "Conditions générales achat", "Avenant contrat logistique",
                         "Appel d'offres nettoyage", "Bon de commande 2024", "Litige fournisseur transport"],
        "routing": ["Car sharing instance generator", "Vehicle routing notes", "Relocation heuristic",
                    "Benchmark results", "Time windows model", "Fleet sizing study"],
    }
    makers = ["docx", "pdf", "docx", "xlsx", "pptx", "txt"]
    for topic, names in titles.items():
        for i, name in enumerate(names):
            body = paragraph(topic, rng, 8)
            kind = makers[i % len(makers)]
            folder = rng.choice([docs, dl, desk])
            safe = name.replace("'", " ")
            if kind == "docx":
                make_docx(folder / f"{safe}.docx", name, body)
            elif kind == "pdf":
                make_pdf(folder / f"{safe}.pdf", name, body)
            elif kind == "xlsx":
                make_xlsx(folder / f"{safe}.xlsx", name, topic, rng)
            elif kind == "pptx":
                make_pptx(folder / f"{safe}.pptx", name, topic, rng)
            else:
                (folder / f"{safe}.txt").write_text(f"{name}\n\n{body}", encoding="utf-8")
        make_eml(dl / f"RE {titles[topic][0]}.eml", f"RE: {titles[topic][0]}", paragraph(topic, rng, 5),
                 attach=f"{titles[topic][1]}.pdf")

    # duplicates with different names / formats
    shutil.copy(docs.parent / "Documents" / "Brand guidelines v2.docx" if (docs / "Brand guidelines v2.docx").exists()
                else next(root.rglob("Brand guidelines v2.*")), desk / "brand guidelines - Copie.docx")
    src_pdf = next(root.rglob("*.pdf"))
    shutil.copy(src_pdf, dl / "scan_0001.pdf")
    first_txt = next(root.rglob("*.txt"))
    (dl / "notes (2).md").write_text(first_txt.read_text(encoding="utf-8").replace("\n\n", "\n"), encoding="utf-8")

    # team-building photos: two events on different days + one resized duplicate
    ev1 = pics / "Séminaire Annecy 2024"
    ev1.mkdir()
    for k in range(6):
        make_photo(ev1 / f"IMG_{4100 + k}.JPG", f"2024:06:14 1{k}:12:00", seed=k)
    ev2 = pics / "WhatsApp Images"
    ev2.mkdir()
    for k in range(5):
        make_photo(ev2 / f"IMG-20241219-WA00{k}.jpg", f"2024:12:19 2{k % 3}:05:00", seed=50 + k)
    from PIL import Image

    with Image.open(ev1 / "IMG_4101.JPG") as im:
        exif = im.info.get("exif")
        im.resize((800, 600)).save(dl / "photo equipe.jpg", quality=80, exif=exif)

    # screenshots with text, logos, a screen recording
    shots = pics / "Screenshots"
    shots.mkdir()
    make_screenshot(shots / "Screenshot 2024-05-02 101500.png",
                    ["Budget previsionnel 2025 - tresorerie", "Charges de personnel: 1 250 000",
                     "Investissements: 300 000", "Ecart budgetaire T1: -4%", "Comite de direction: validation"], 1)
    make_screenshot(shots / "Capture d'écran 2024-05-03.png",
                    ["Phishing awareness training", "Do not click suspicious links", "Report suspicious email",
                     "Enable multi-factor authentication", "Contact IT helpdesk"], 2)
    for k, c in enumerate([(220, 30, 30), (30, 120, 220), (30, 160, 80)]):
        make_logo(desk / f"logo_v{k + 1}.png", k, c)
    make_video(desk / "Screen Recording 2024-06-01.mp4")

    # things that can't be sorted
    (dl / "setup_tool.exe").write_bytes(b"MZ" + bytes(range(256)) * 40)
    (dl / "data.bin").write_bytes(bytes(rng.getrandbits(8) for _ in range(5000)))
    (dl / "broken.pdf").write_bytes(b"%PDF-1.4 this is not really a pdf")
    (dl / "empty.txt").write_text("")
    return root


if __name__ == "__main__":
    out = build(Path(sys.argv[1] if len(sys.argv) > 1 else "sample"))
    print(f"sample created in {out}")
