"""Reads the content of files: text, metadata and preview images. Everything is parsed locally.

Each extractor is defensive: a broken or password-protected file returns an `Extracted` with `error` set instead
of raising, so one bad file never stops a run.
"""

from __future__ import annotations

import email
import email.policy
import html
import json
import logging
import re
import tarfile
import threading
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from pathlib import Path

from .paths import long_path
from .textutil import normalize_spaces

log = logging.getLogger(__name__)

MAX_TEXT_CHARS = 60_000          # text kept per file (enough to understand its subject)
MAX_TEXT_BYTES = 2_000_000       # bytes read from plain-text files
MAX_PDF_PAGES = 12
MAX_SHEET_ROWS = 300
PREVIEW_SIZE = 640               # longest side of preview images kept for visual analysis

KIND_FOLDERS = {
    "document": "Documents",
    "spreadsheet": "Spreadsheets",
    "presentation": "Presentations",
    "email": "Emails",
    "image": "Images",
    "design": "Design files",
    "video": "Videos",
    "audio": "Audio",
    "archive": "Archives",
    "code": "Code",
    "program": "Programs & installers",
    "other": "Other files",
}

_EXT = {}
for _kind, _exts in {
    "document": ".txt .md .markdown .rst .log .pdf .docx .docm .dotx .dotm .doc .dot .odt .rtf .tex .html .htm .xhtml"
                " .mht .mhtml .epub .xml .json .yaml .yml .toml .cfg .conf .ini .srt .vtt .pages",
    "spreadsheet": ".xlsx .xlsm .xltx .xltm .xls .xlsb .ods .csv .tsv .numbers",
    "presentation": ".pptx .pptm .ppsx .potx .ppt .pps .odp .key",
    "email": ".eml .msg .emlx",
    "image": ".jpg .jpeg .jfif .png .gif .bmp .tif .tiff .webp .heic .heif .avif",
    "design": ".psd .ai .svg .eps .indd .idml .fig .sketch .xd .cdr .afdesign .afphoto .dwg .dxf .skp",
    "video": ".mp4 .m4v .mov .avi .mkv .wmv .webm .mpg .mpeg .3gp .flv .mts .m2ts",
    "audio": ".mp3 .wav .m4a .aac .flac .ogg .oga .opus .wma .aiff .amr",
    "archive": ".zip .7z .rar .tar .gz .tgz .bz2 .xz .cab .iso",
    "code": ".py .ipynb .js .jsx .ts .tsx .java .kt .c .h .cpp .hpp .cs .go .rs .rb .php .sql .sh .ps1 .bat .cmd"
            " .vb .vbs .bas .r .m .swift .scala .css .scss .less .lua .pl",
    "program": ".exe .msi .msix .appx .apk .dmg .pkg .deb .rpm .jar",
}.items():
    for _e in _exts.split():
        _EXT[_e] = _kind

PLAIN_TEXT_EXT = {
    ".txt", ".md", ".markdown", ".rst", ".log", ".csv", ".tsv", ".xml", ".json", ".yaml", ".yml", ".toml", ".cfg",
    ".conf", ".ini", ".srt", ".vtt", ".tex",
} | {e for e, k in _EXT.items() if k == "code" and e != ".ipynb"}

_PDFIUM_LOCK = threading.Lock()   # PDFium is not thread-safe
_OPTIONAL_LOGGED: set[str] = set()

try:  # iPhone photos (HEIC) support, if installed
    from pillow_heif import register_heif_opener

    register_heif_opener()
except Exception:  # pragma: no cover - optional
    pass


def kind_of(path: Path) -> str:
    return _EXT.get(path.suffix.lower(), "other")


@dataclass
class Extracted:
    kind: str
    text: str = ""
    title: str = ""
    meta: dict = field(default_factory=dict)
    images: list = field(default_factory=list)   # PIL images (the picture itself, video frames, a PDF page...)
    error: str | None = None
    needs_ocr: bool = False                       # images that probably contain text (scans, screenshots)


def extract(path: Path) -> Extracted:
    kind = kind_of(path)
    ext = path.suffix.lower()
    try:
        if ext == ".pdf" or ext == ".ai":
            res = _pdf(path, kind)
        elif ext in (".docx", ".docm", ".dotx", ".dotm"):
            res = _docx(path)
        elif ext in (".xlsx", ".xlsm", ".xltx", ".xltm"):
            res = _xlsx(path)
        elif ext == ".xls":
            res = _xls(path)
        elif ext in (".pptx", ".pptm", ".ppsx", ".potx"):
            res = _pptx(path)
        elif ext in (".odt", ".ods", ".odp"):
            res = _opendocument(path, kind)
        elif ext in (".doc", ".dot", ".ppt", ".pps", ".xlsb"):
            res = Extracted(kind, text=_binary_strings(path))
        elif ext == ".rtf":
            res = _rtf(path)
        elif ext in (".html", ".htm", ".xhtml", ".mht", ".mhtml"):
            res = _html_file(path)
        elif ext == ".epub":
            res = _epub(path)
        elif ext == ".ipynb":
            res = _ipynb(path)
        elif ext == ".svg":
            res = _svg(path)
        elif ext in PLAIN_TEXT_EXT:
            res = Extracted(kind, text=_read_text(path))
        elif ext in (".eml", ".emlx"):
            res = _eml(path)
        elif ext == ".msg":
            res = _msg(path)
        elif kind == "image" or ext == ".psd":
            res = _image(path, "image" if kind == "image" else kind)
        elif kind == "video":
            res = _video(path)
        elif kind == "audio":
            res = _audio(path)
        elif ext in (".zip", ".tar", ".tgz", ".gz", ".bz2", ".xz"):
            res = _archive(path)
        else:
            res = Extracted(kind)
    except MemoryError:
        res = Extracted(kind, error="file too large to read")
    except Exception as exc:  # noqa: BLE001 - never let one file stop the run
        msg = str(exc).strip() or exc.__class__.__name__
        if "password" in msg.lower() or "encrypt" in msg.lower():
            msg = "password protected"
        res = Extracted(kind, error=msg[:200])
    res.kind = res.kind or kind
    res.text = normalize_spaces(res.text)[:MAX_TEXT_CHARS]
    res.title = normalize_spaces(res.title)[:200]
    return res


# ---------------------------------------------------------------------------------------------------------------
# Plain text, HTML, RTF
# ---------------------------------------------------------------------------------------------------------------
def decode_bytes(data: bytes) -> str:
    if data.startswith(b"\xef\xbb\xbf"):
        return data[3:].decode("utf-8", errors="replace")
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16", errors="replace")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("cp1252", errors="replace")


def _read_text(path: Path, limit: int = MAX_TEXT_BYTES) -> str:
    with open(long_path(path), "rb") as f:
        return decode_bytes(f.read(limit))


class _HTMLText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title = ""
        self._skip = 0
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "noscript", "svg"):
            self._skip += 1
        elif tag == "title":
            self._in_title = True
        elif tag in ("p", "br", "div", "li", "tr", "h1", "h2", "h3", "h4", "td"):
            self.parts.append(" ")

    def handle_endtag(self, tag):
        if tag in ("script", "style", "noscript", "svg") and self._skip:
            self._skip -= 1
        elif tag == "title":
            self._in_title = False

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif not self._skip:
            self.parts.append(data)


def html_to_text(markup: str) -> tuple[str, str]:
    p = _HTMLText()
    try:
        p.feed(markup)
        p.close()
    except Exception:  # noqa: BLE001
        pass
    return normalize_spaces(" ".join(p.parts)), normalize_spaces(p.title)


def _html_file(path: Path) -> Extracted:
    raw = _read_text(path)
    if path.suffix.lower() in (".mht", ".mhtml"):
        msg = email.message_from_string(raw, policy=email.policy.default)
        parts = [p.get_content() for p in msg.walk() if p.get_content_type() in ("text/html", "text/plain")]
        raw = " ".join(x for x in parts if isinstance(x, str)) or raw
    text, title = html_to_text(raw)
    return Extracted("document", text=text, title=title)


def _rtf(path: Path) -> Extracted:
    from striprtf.striprtf import rtf_to_text

    return Extracted("document", text=rtf_to_text(_read_text(path), errors="ignore"))


def _binary_strings(path: Path, limit: int = 6_000_000) -> str:
    """Best-effort text from legacy binary Office files (.doc/.ppt): UTF-16 and 8-bit runs of readable text."""
    with open(long_path(path), "rb") as f:
        data = f.read(limit)
    utf16 = [m.decode("utf-16-le", errors="ignore") for m in re.findall(rb"(?:[\x20-\x7e\xa0-\xff]\x00){5,}", data)]
    ansi = [m.decode("cp1252", errors="ignore") for m in re.findall(rb"[\x20-\x7e\xa0-\xff]{8,}", data)]
    best = utf16 if sum(map(len, utf16)) >= sum(map(len, ansi)) else ansi

    def readable(s: str) -> bool:
        letters = sum(c.isalpha() for c in s)
        return " " in s and letters >= 0.6 * len(s.replace(" ", ""))

    return " ".join(s for s in best if readable(s))


# ---------------------------------------------------------------------------------------------------------------
# PDF (and Illustrator files saved with PDF compatibility)
# ---------------------------------------------------------------------------------------------------------------
def _pdf(path: Path, kind: str) -> Extracted:
    import pypdfium2 as pdfium

    with _PDFIUM_LOCK:
        pdf = pdfium.PdfDocument(long_path(path))
        try:
            n = len(pdf)
            texts = []
            for i in range(min(n, MAX_PDF_PAGES)):
                page = pdf[i]
                tp = page.get_textpage()
                texts.append(tp.get_text_range())
                tp.close()
                page.close()
            try:
                md = pdf.get_metadata_dict(skip_empty=True)
            except Exception:  # noqa: BLE001
                md = {}
            text = " ".join(texts)
            images = []
            needs_ocr = False
            if n and (len(text.strip()) < 80 or kind == "design"):
                # Scanned PDF or design file: keep a picture of the first page.
                page = pdf[0]
                w, h = page.get_size()
                scale = min(3.0, 1600 / max(w, h, 1))
                images.append(page.render(scale=scale).to_pil().convert("RGB"))
                page.close()
                needs_ocr = kind != "design"
        finally:
            pdf.close()
    meta = {"pages": n}
    if md.get("Author"):
        meta["author"] = md["Author"]
    created = _pdf_date(md.get("CreationDate", ""))
    if created:
        meta["created"] = created
    return Extracted(kind, text=text, title=md.get("Title", ""), meta=meta, images=images, needs_ocr=needs_ocr)


def _pdf_date(s: str) -> float | None:
    m = re.match(r"D?:?(\d{4})(\d{2})?(\d{2})?(\d{2})?(\d{2})?", s or "")
    if not m:
        return None
    try:
        parts = [int(x) if x else d for x, d in zip(m.groups(), (0, 1, 1, 0, 0))]
        return datetime(*parts).timestamp()
    except (ValueError, OverflowError, OSError):
        return None


# ---------------------------------------------------------------------------------------------------------------
# Office Open XML and OpenDocument
# ---------------------------------------------------------------------------------------------------------------
def _core_props(props) -> tuple[str, dict]:
    meta = {}
    try:
        if props.author:
            meta["author"] = props.author
        if props.created:
            meta["created"] = props.created.timestamp()
        return props.title or "", meta
    except Exception:  # noqa: BLE001
        return "", meta


def _docx(path: Path) -> Extracted:
    import docx

    d = docx.Document(long_path(path))
    parts = [p.text for p in d.paragraphs if p.text]
    for t in d.tables[:30]:
        for row in t.rows[:100]:
            for cell in row.cells:
                if cell.text:
                    parts.append(cell.text)
        if sum(map(len, parts)) > MAX_TEXT_CHARS:
            break
    title, meta = _core_props(d.core_properties)
    return Extracted("document", text=" ".join(parts), title=title, meta=meta)


def _xlsx(path: Path) -> Extracted:
    import openpyxl

    wb = openpyxl.load_workbook(long_path(path), read_only=True, data_only=True)
    try:
        parts = []
        sheets = []
        for ws in wb.worksheets[:12]:
            sheets.append(ws.title)
            parts.append(ws.title)
            for row in ws.iter_rows(max_row=MAX_SHEET_ROWS, values_only=True):
                parts.extend(str(v) for v in row if isinstance(v, str) and v.strip())
                if sum(map(len, parts)) > MAX_TEXT_CHARS:
                    break
        title, meta = _core_props(wb.properties)
    finally:
        wb.close()
    meta["sheets"] = sheets
    return Extracted("spreadsheet", text=" ".join(parts), title=title, meta=meta)


def _xls(path: Path) -> Extracted:
    import xlrd

    book = xlrd.open_workbook(long_path(path), on_demand=True)
    try:
        parts = []
        for i in range(min(book.nsheets, 12)):
            sh = book.sheet_by_index(i)
            parts.append(sh.name)
            for r in range(min(sh.nrows, MAX_SHEET_ROWS)):
                parts.extend(str(v) for v in sh.row_values(r) if isinstance(v, str) and v.strip())
    finally:
        book.release_resources()
    return Extracted("spreadsheet", text=" ".join(parts))


def _pptx(path: Path) -> Extracted:
    from pptx import Presentation

    prs = Presentation(long_path(path))
    parts = []
    for slide in prs.slides:
        for shape in slide.shapes:
            if getattr(shape, "has_text_frame", False) and shape.has_text_frame:
                parts.append(shape.text_frame.text)
            if getattr(shape, "has_table", False) and shape.has_table:
                for row in shape.table.rows:
                    parts.extend(c.text for c in row.cells if c.text)
        if slide.has_notes_slide:
            parts.append(slide.notes_slide.notes_text_frame.text if slide.notes_slide.notes_text_frame else "")
        if sum(map(len, parts)) > MAX_TEXT_CHARS:
            break
    title, meta = _core_props(prs.core_properties)
    meta["slides"] = len(prs.slides)
    return Extracted("presentation", text=" ".join(parts), title=title, meta=meta)


def _strip_xml(xml: str) -> str:
    xml = re.sub(r"<(text:p|text:h|table:table-cell|text:line-break)[^>]*>", " ", xml)
    return html.unescape(re.sub(r"<[^>]+>", "", xml))


def _opendocument(path: Path, kind: str) -> Extracted:
    with zipfile.ZipFile(long_path(path)) as z:
        content = z.read("content.xml").decode("utf-8", errors="replace")
        title = ""
        if "meta.xml" in z.namelist():
            m = re.search(r"<dc:title>(.*?)</dc:title>", z.read("meta.xml").decode("utf-8", errors="replace"), re.S)
            title = html.unescape(m.group(1)) if m else ""
    return Extracted(kind, text=_strip_xml(content), title=title)


def _epub(path: Path) -> Extracted:
    parts = []
    title = ""
    with zipfile.ZipFile(long_path(path)) as z:
        for name in z.namelist():
            if name.lower().endswith((".xhtml", ".html", ".htm")):
                text, t = html_to_text(z.read(name).decode("utf-8", errors="replace"))
                parts.append(text)
                title = title or t
            elif name.lower().endswith(".opf"):
                m = re.search(r"<dc:title[^>]*>(.*?)</dc:title>", z.read(name).decode("utf-8", errors="replace"))
                if m:
                    title = html.unescape(m.group(1))
            if sum(map(len, parts)) > MAX_TEXT_CHARS:
                break
    return Extracted("document", text=" ".join(parts), title=title)


def _ipynb(path: Path) -> Extracted:
    nb = json.loads(_read_text(path))
    cells = nb.get("cells", [])
    text = " ".join("".join(c.get("source", "")) if isinstance(c.get("source"), list) else str(c.get("source", ""))
                    for c in cells)
    return Extracted("code", text=text)


def _svg(path: Path) -> Extracted:
    raw = _read_text(path)
    texts = re.findall(r"<(?:title|desc|text|tspan)[^>]*>([^<]+)<", raw)
    return Extracted("design", text=html.unescape(" ".join(texts)))


# ---------------------------------------------------------------------------------------------------------------
# E-mails
# ---------------------------------------------------------------------------------------------------------------
def _eml(path: Path) -> Extracted:
    with open(long_path(path), "rb") as f:
        data = f.read(MAX_TEXT_BYTES * 5)
    if path.suffix.lower() == ".emlx":  # Apple Mail: first line holds the byte count
        data = data.split(b"\n", 1)[1] if b"\n" in data else data
    msg = email.message_from_bytes(data, policy=email.policy.default)
    body = ""
    try:
        part = msg.get_body(preferencelist=("plain", "html"))
        if part is not None:
            body = part.get_content()
            if part.get_content_type() == "text/html":
                body = html_to_text(body)[0]
    except Exception:  # noqa: BLE001
        pass
    attachments = []
    try:
        attachments = [a.get_filename() or "" for a in msg.iter_attachments()]
    except Exception:  # noqa: BLE001
        pass
    meta = {"from": str(msg.get("From", ""))[:200], "to": str(msg.get("To", ""))[:200]}
    try:
        meta["created"] = parsedate_to_datetime(str(msg.get("Date"))).timestamp()
    except Exception:  # noqa: BLE001
        pass
    if attachments:
        meta["attachments"] = [a for a in attachments if a][:50]
    subject = str(msg.get("Subject", "") or "")
    text = f"{subject} {body} {' '.join(meta.get('attachments', []))}"
    return Extracted("email", text=text, title=subject, meta=meta)


def _msg(path: Path) -> Extracted:
    import extract_msg

    m = extract_msg.openMsg(long_path(path))
    try:
        subject = m.subject or ""
        body = m.body or ""
        if not body and getattr(m, "htmlBody", None):
            body = html_to_text(decode_bytes(m.htmlBody))[0]
        attachments = []
        for a in getattr(m, "attachments", []) or []:
            name = getattr(a, "longFilename", None) or getattr(a, "shortFilename", None) or ""
            if name:
                attachments.append(name)
        meta = {"from": str(m.sender or "")[:200], "to": str(m.to or "")[:200]}
        try:
            if m.date:
                meta["created"] = m.date.timestamp()
        except Exception:  # noqa: BLE001
            pass
    finally:
        m.close()
    if attachments:
        meta["attachments"] = attachments[:50]
    text = f"{subject} {body} {' '.join(attachments)}"
    return Extracted("email", text=text, title=subject, meta=meta)


# ---------------------------------------------------------------------------------------------------------------
# Images, videos, audio
# ---------------------------------------------------------------------------------------------------------------
_SCREEN_SIZES = {
    (1280, 720), (1280, 800), (1280, 1024), (1366, 768), (1440, 900), (1536, 864), (1600, 900), (1680, 1050),
    (1920, 1080), (1920, 1200), (2048, 1152), (2560, 1080), (2560, 1440), (2560, 1600), (2880, 1800), (3440, 1440),
    (3840, 2160), (2736, 1824), (3000, 2000), (2256, 1504), (1024, 768), (800, 600), (2880, 1620), (3024, 1964),
}
_SCREEN_NAME = re.compile(
    r"(screen ?shot|screenshot|capture|capture d.?[ée]cran|snip|snipping|bildschirm|schermata|captura|"
    r"screen ?recording|enregistrement|recording|obs|bandicam|loom|teams|zoom|webex|^\d{4}-\d{2}-\d{2} \d{2}-\d{2}-\d{2})",
    re.IGNORECASE,
)


def looks_like_screen(name: str, width: int, height: int, has_camera: bool) -> bool:
    if has_camera:
        return False
    if _SCREEN_NAME.search(name):
        return True
    w, h = max(width, height), min(width, height)
    return (w, h) in _SCREEN_SIZES


def _exif_info(img) -> dict:
    meta = {}
    try:
        exif = img.getexif()
    except Exception:  # noqa: BLE001
        return meta
    if not exif:
        return meta
    try:
        sub = exif.get_ifd(0x8769)
    except Exception:  # noqa: BLE001
        sub = {}
    dt = sub.get(36867) or sub.get(36868) or exif.get(306)
    if isinstance(dt, str):
        try:
            meta["taken"] = datetime.strptime(dt.strip()[:19], "%Y:%m:%d %H:%M:%S").timestamp()
        except (ValueError, OverflowError, OSError):
            pass
    make, model = exif.get(271), exif.get(272)
    if make or model:
        meta["camera"] = normalize_spaces(f"{make or ''} {model or ''}")[:80]
    if exif.get(305):
        meta["software"] = str(exif.get(305))[:80]
    try:
        if exif.get_ifd(0x8825):
            meta["gps"] = True
    except Exception:  # noqa: BLE001
        pass
    return meta


def _image(path: Path, kind: str) -> Extracted:
    from PIL import Image, ImageOps

    Image.MAX_IMAGE_PIXELS = 400_000_000
    with Image.open(long_path(path)) as img:
        width, height = img.size
        meta = {"width": width, "height": height, "format": img.format or ""}
        meta.update(_exif_info(img))
        if isinstance(img.info.get("Software"), str):
            meta.setdefault("software", img.info["Software"][:80])
        try:
            img.draft("RGB", (PREVIEW_SIZE * 2, PREVIEW_SIZE * 2))  # fast reduced-size JPEG decoding
        except Exception:  # noqa: BLE001
            pass
        if getattr(img, "n_frames", 1) > 1:
            img.seek(0)
        im = ImageOps.exif_transpose(img)
        if im.mode in ("RGBA", "LA", "P"):
            im = im.convert("RGBA")
            bg = Image.new("RGB", im.size, (255, 255, 255))
            bg.paste(im, mask=im.split()[-1])
            im = bg
        else:
            im = im.convert("RGB")
        im.thumbnail((PREVIEW_SIZE, PREVIEW_SIZE), Image.Resampling.LANCZOS)
    screen = looks_like_screen(path.name, width, height, "camera" in meta)
    meta["screen"] = screen
    needs_ocr = screen or (not meta.get("camera") and path.suffix.lower() in (".png", ".bmp", ".gif", ".tif", ".tiff"))
    return Extracted(kind, meta=meta, images=[im], needs_ocr=needs_ocr)


def _video(path: Path) -> Extracted:
    try:
        import av
    except ImportError:
        return Extracted("video", error="video support not installed")

    frames = []
    meta: dict = {}
    with av.open(long_path(path)) as container:
        if not container.streams.video:
            return _audio(path)
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        duration = 0.0
        if container.duration:
            duration = container.duration / 1_000_000
        elif stream.duration and stream.time_base:
            duration = float(stream.duration * stream.time_base)
        meta.update(duration=round(duration, 2), width=stream.codec_context.width, height=stream.codec_context.height)
        tags = {k.lower(): v for k, v in (container.metadata or {}).items()}
        created = tags.get("creation_time") or tags.get("date")
        if created:
            try:
                meta["taken"] = datetime.fromisoformat(created.replace("Z", "+00:00")[:25]).timestamp()
            except ValueError:
                pass
        if tags.get("title"):
            meta["tag_title"] = tags["title"][:200]
        for frac in (0.1, 0.35, 0.6, 0.85):
            try:
                if duration > 0:
                    container.seek(int(duration * frac * 1_000_000), any_frame=False, backward=True)
                for frame in container.decode(stream):
                    im = frame.to_image()
                    im.thumbnail((PREVIEW_SIZE, PREVIEW_SIZE))
                    frames.append(im.convert("RGB"))
                    break
            except Exception:  # noqa: BLE001
                continue
            if duration <= 0:
                break
    if not frames:
        return Extracted("video", meta=meta, error="could not decode video frames")
    screen = looks_like_screen(path.name, meta.get("width", 0), meta.get("height", 0), False)
    meta["screen"] = screen
    return Extracted("video", title=meta.get("tag_title", ""), meta=meta, images=frames, needs_ocr=screen)


def _audio(path: Path) -> Extracted:
    meta: dict = {}
    text = ""
    try:
        import av

        with av.open(long_path(path)) as c:
            if c.duration:
                meta["duration"] = round(c.duration / 1_000_000, 2)
            tags = {k.lower(): v for k, v in (c.metadata or {}).items()}
            text = " ".join(str(tags[k]) for k in ("title", "album", "artist", "comment", "genre") if tags.get(k))
    except Exception:  # noqa: BLE001
        pass
    return Extracted("audio", text=text, meta=meta)


def _archive(path: Path) -> Extracted:
    names: list[str] = []
    p = long_path(path)
    if zipfile.is_zipfile(p):
        with zipfile.ZipFile(p) as z:
            names = z.namelist()[:500]
    elif tarfile.is_tarfile(p):
        with tarfile.open(p) as t:
            for i, m in enumerate(t):
                names.append(m.name)
                if i > 500:
                    break
    words = " ".join(re.sub(r"[/\\_.\-]+", " ", n) for n in names)
    return Extracted("archive", text=words, meta={"entries": len(names)})
