# PyInstaller spec – builds dist/LocalFileSorter/LocalFileSorter.exe (folder build: starts fast, no unpacking).
# Usage (from the repository root):  pyinstaller packaging/LocalFileSorter.spec --noconfirm
# Models are NOT bundled here; the build workflow copies the `models` folder next to the .exe.

import os
import sys

from PyInstaller.utils.hooks import collect_all

ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))
sys.path.insert(0, ROOT)

datas, binaries, hiddenimports = [], [], []
for pkg in ("rapidocr_onnxruntime", "pypdfium2", "pypdfium2_raw", "onnxruntime", "tokenizers", "av",
            "extract_msg", "pillow_heif", "llama_cpp", "docx", "pptx", "openpyxl"):
    try:
        d, b, h = collect_all(pkg)
        datas += d
        binaries += b
        hiddenimports += h
    except Exception as exc:  # optional packages may be missing
        print(f"[spec] skipping {pkg}: {exc}")

hiddenimports += [
    "sklearn.utils._typedefs", "sklearn.utils._heap", "sklearn.utils._sorting", "sklearn.utils._vector_sentinel",
    "sklearn.neighbors._partition_nodes", "sklearn.cluster._agglomerative", "sklearn.cluster._kmeans",
    "watchdog.observers.read_directory_changes", "watchdog.observers.polling",
]

icon_path = os.path.join(SPECPATH, "LocalFileSorter.ico")
if not os.path.exists(icon_path):
    from filesorter.icon import save_ico

    save_ico(icon_path)

a = Analysis(
    [os.path.join(SPECPATH, "launcher.py")],
    pathex=[ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["tkinter", "matplotlib", "torch", "tensorflow", "IPython", "pytest", "onnx", "fpdf",
              "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.QtQuick", "PySide6.QtQml",
              "PySide6.Qt3DCore", "PySide6.QtMultimedia", "PySide6.QtPdf"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="LocalFileSorter",
    console=False,
    icon=icon_path,
    upx=False,
)
coll = COLLECT(exe, a.binaries, a.datas, name="LocalFileSorter", upx=False)
