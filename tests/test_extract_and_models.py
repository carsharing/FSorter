import numpy as np
from PIL import Image

from filesorter.extract import extract
from filesorter.models import (BasicVisualEmbedder, ClipModel, ModelHub, OnnxTextEmbedder, TfidfTextEmbedder,
                               clean_llm_name)


def test_extract_office_formats(sample):
    by_name = {p.name: p for p in sample.rglob("*") if p.is_file()}
    docx = extract(by_name["Onboarding checklist.docx"])
    assert docx.kind == "document" and "onboarding" in docx.text.lower() and docx.title == "Onboarding checklist"
    xlsx = extract(next(p for n, p in by_name.items() if n.endswith(".xlsx")))
    assert xlsx.kind == "spreadsheet" and xlsx.text and xlsx.meta["sheets"]
    pdf = extract(next(p for n, p in by_name.items() if n.endswith(".pdf") and n != "broken.pdf"))
    assert pdf.kind == "document" and len(pdf.text) > 100 and pdf.meta["pages"] >= 1
    pptx = extract(next(p for n, p in by_name.items() if n.endswith(".pptx")))
    assert pptx.kind == "presentation" and pptx.meta["slides"] == 4 and pptx.text
    eml = extract(next(p for n, p in by_name.items() if n.endswith(".eml")))
    assert eml.kind == "email" and eml.title.startswith("RE:") and eml.meta.get("attachments")
    broken = extract(by_name["broken.pdf"])
    assert broken.error


def test_extract_photo_and_video(sample):
    photo = extract(next(sample.rglob("IMG_4100.JPG")))
    assert photo.kind == "image" and photo.meta["camera"] == "Apple iPhone 13" and photo.meta["taken"]
    assert photo.images and max(photo.images[0].size) <= 640
    video = extract(next(sample.rglob("*.mp4")))
    assert video.kind == "video" and video.error is None and len(video.images) >= 2
    assert video.meta["duration"] > 0 and video.meta["screen"]


def test_onnx_text_model_plumbing(tmp_path):
    import dummy_models

    folder = dummy_models.make_text_model(tmp_path / "text")
    emb = OnnxTextEmbedder(folder, threads=1)
    V = emb.embed(["supplier invoice payment", "fournisseur facture paiement", "team party photo", "x" * 5000])
    assert V.shape == (4, 16)
    assert np.allclose(np.linalg.norm(V, axis=1), 1, atol=1e-5)
    assert V[0] @ V[1] > V[0] @ V[2]  # cross-language synonyms are close in the dummy model


def test_onnx_clip_plumbing(tmp_path):
    import dummy_models

    folder = dummy_models.make_clip_model(tmp_path / "clip")
    clip = ClipModel(folder, threads=1)
    imgs = [Image.new("RGB", (300, 200), c) for c in ("red", "blue")]
    V = clip.embed_images(imgs)
    assert V.shape == (2, 8)
    L = clip.label_embeddings()
    assert L.shape[1] == 8 and L.shape[0] > 50


def test_model_hub_falls_back_when_models_missing(tmp_path):
    hub = ModelHub(tmp_path / "nothing", threads=1, ocr=False, ai_naming=True)
    assert isinstance(hub.text, TfidfTextEmbedder)
    assert isinstance(hub.visual, BasicVisualEmbedder)
    assert hub.namer is None
    st = hub.status()
    assert not st["text_model_ok"] and not st["image_model_ok"]


def test_model_hub_uses_installed_models(tmp_path):
    import dummy_models

    dummy_models.make_text_model(tmp_path / "models" / "text")
    dummy_models.make_clip_model(tmp_path / "models" / "clip")
    hub = ModelHub(tmp_path / "models", threads=1, ocr=False)
    assert isinstance(hub.text, OnnxTextEmbedder) and isinstance(hub.visual, ClipModel)


def test_clean_llm_name():
    assert clean_llm_name('"Notes de frais"') == "Notes de frais"
    assert clean_llm_name("Folder name: Q3 sales forecasts.") == "Q3 sales forecasts"
    assert clean_llm_name("<think>hmm</think>\nOnboarding new hires") == "Onboarding new hires"
    assert clean_llm_name("This is a long explanation that is clearly not a folder name at all") is None
    assert clean_llm_name("VehicleRoutingStudy") == "Vehicle Routing Study"
    assert clean_llm_name("iPhone") == "iPhone" and clean_llm_name("VBA") == "VBA"
