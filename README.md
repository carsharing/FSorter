# Local File Sorter

A Windows desktop app that reads the **content** of your files (documents, e-mails, spreadsheets, slides, PDFs,
photos, screenshots, screen recordings…) and **copies** them into precise, automatically named folders, with
duplicates set aside, an HTML report of where everything went, and a background mode that keeps sorting new
files. **It works fully offline:** every model runs on your computer and the app blocks all network access.

- Folders are named after real subjects found in your files, for example `Finance/Notes de frais`,
  `Training/Formation Excel – tableaux croisés`, `Photos & events/2024-06-14 Séminaire Annecy`,
  `Screenshots/Spreadsheets`. There is no fixed list of categories.
- It understands 100+ languages, and French and English files about the same subject end up together.
- Duplicates are found by content, even with different names, formats or picture sizes. They go into a
  `duplicate` subfolder of the folder the app chose.
- Files it can't understand go to `_Unsorted`, by type.
- Originals are never touched: everything is **copied**.
- No administrator rights are needed, neither to install nor to run it.

End-user instructions: [docs/USER-GUIDE.md](docs/USER-GUIDE.md).

## How it works

```
scan ─▶ read content ─▶ find duplicates ─▶ understand ─▶ group & name ─▶ copy ─▶ HTML report
```

| Step | What happens | Tech (all local) |
|---|---|---|
| Read | Text from PDF, Word, Excel, PowerPoint, OpenDocument, RTF, HTML, e-mails (`.eml`/`.msg`), legacy `.doc/.xls/.ppt`, archives' file lists. Pictures: EXIF date/camera. Videos: 4 frames + date. Path and file names count as content too. | pypdfium2, python-docx, openpyxl, python-pptx, extract-msg, Pillow (+HEIC), PyAV |
| OCR | Text in screenshots, scanned PDFs and screen recordings, which are then sorted by subject. | RapidOCR (ONNX, models inside the package) |
| Duplicates | Identical bytes (size → quick hash → full hash), same text (normalised fingerprint), look-alike pictures (pHash + dHash + colour check), same videos (frame hashes + duration). | numpy |
| Understand documents | Multilingual sentence embeddings. | `multilingual-e5-base` (ONNX int8) via onnxruntime |
| Understand pictures | Image embeddings + zero-shot recognition of ~90 office-life concepts (team building, receipts, whiteboards, spreadsheets on screen, logos…). | CLIP ViT-B/16 (ONNX) |
| Group | Average-linkage clustering with thresholds measured on *your* collection; big groups split; two levels (area → precise topic). Event photos are grouped by date. | scikit-learn |
| Name | Distinctive keywords and phrases per group (class-based TF-IDF), keeping the original spelling ("VPN", "chiffre d'affaires"). Optionally a small local LLM turns them into natural names. | Qwen2.5-1.5B-Instruct (GGUF) via llama.cpp |
| Copy | Writes to a temporary name and then renames (no half-copied files). Handles name clashes, long paths and low disk space. | |
| Report | A single self-contained HTML file: folder tree, keywords, a searchable/filterable table of every file, CSV export. | vanilla JS |
| Background | Watches the folders. New files join existing folders, a new subject creates a new folder, and changed files are re-copied. Runs at low priority and starts with Windows (per-user). | watchdog, HKCU Run key |

**Offline guarantee.** The app contains no download or network code. `filesorter/offline.py` patches Python's
socket layer at start-up so any non-loopback connection or DNS lookup fails. `selftest` checks this. Models are
read from the `models` folder next to the `.exe`. When a model is missing, built-in fallbacks are used (TF-IDF for
text, colour/shape features for images), so nothing is ever fetched.

## Get the Windows app

The **Build Windows app** GitHub Action builds everything on a Windows runner. It downloads the models on the
build machine, runs the tests, self-tests the packaged `.exe` with the real models, and sorts a sample folder:

- Run it from the Actions tab ("Run workflow"), then download the `LocalFileSorter-windows` and
  `LocalFileSorter-naming-model` artifacts.
- Or push a tag (`git tag v0.1.0 && git push --tags`) to get a GitHub Release with both zips attached.

Unzip both into the same folder and run `LocalFileSorter\LocalFileSorter.exe`.

## Develop

```bash
python -m venv .venv && . .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
pip install --no-deps rapidocr_onnxruntime==1.4.4     # OCR (use 1.2.3 on Python 3.13)
python tools/download_models.py --out models          # optional: real models (needs internet, once)
python -m pytest -q
python tests/make_sample.py sample && python -m filesorter run --source sample --dest sorted
python -m filesorter                                  # the window
```

Build the `.exe` locally: `pip install -r requirements-build.txt && pyinstaller packaging/LocalFileSorter.spec`,
then copy `models/` into `dist/LocalFileSorter/`.

### Code map

| File | Role |
|---|---|
| `filesorter/engine.py` | the pipeline: scan, analyse (incremental), duplicates, first-run organisation, later-run assignment, copy |
| `filesorter/organize.py` | clustering, two-level hierarchy, event grouping, folder naming |
| `filesorter/extract.py` | content extraction for every supported format |
| `filesorter/models.py` | ONNX text/image models, OCR, naming LLM, fallbacks |
| `filesorter/features.py` | hashes and fingerprints for duplicate detection |
| `filesorter/report.py` | the HTML report |
| `filesorter/gui.py`, `background.py`, `autostart.py` | window, tray, background watcher, start with Windows |
| `filesorter/offline.py` | network blocking |
| `tools/download_models.py` | build-time model download + sanity checks |
