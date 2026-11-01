"""Tiny stand-in ONNX models with the same inputs/outputs as the real ones, to test the model plumbing
without downloading anything."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _tokenizer(words: list[str], path: Path, specials=("<pad>", "[UNK]"), add_bos_eos=False):
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors

    vocab = {w: i for i, w in enumerate(list(specials) + words)}
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tok.normalizer = normalizers.Lowercase()
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    if add_bos_eos:
        tok.post_processor = processors.TemplateProcessing(
            single="<|startoftext|> $A <|endoftext|>",
            special_tokens=[("<|startoftext|>", vocab["<|startoftext|>"]), ("<|endoftext|>", vocab["<|endoftext|>"])])
    tok.save(str(path))
    return vocab


WORDS = ("invoice facture payment paiement supplier fournisseur budget training formation excel photo team party "
         "logo brand security password phishing spreadsheet screenshot code a of the and with people group").split()


def make_text_model(folder: Path, dim: int = 16) -> Path:
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    folder.mkdir(parents=True, exist_ok=True)
    vocab = _tokenizer(WORDS, folder / "tokenizer.json")
    rng = np.random.default_rng(0)
    E = rng.normal(size=(len(vocab), dim)).astype(np.float32)
    # related words share a direction so similarity is meaningful
    for a, b in [("invoice", "facture"), ("payment", "paiement"), ("supplier", "fournisseur"),
                 ("training", "formation")]:
        E[vocab[b]] = E[vocab[a]] + 0.05 * rng.normal(size=dim)
    graph = helper.make_graph(
        [helper.make_node("Gather", ["E", "input_ids"], ["last_hidden_state"], axis=0)],
        "text",
        [helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["b", "l"]),
         helper.make_tensor_value_info("attention_mask", TensorProto.INT64, ["b", "l"])],
        [helper.make_tensor_value_info("last_hidden_state", TensorProto.FLOAT, ["b", "l", dim])],
        [numpy_helper.from_array(E, "E")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.save(model, str(folder / "model.onnx"))
    (folder / "filesorter_model.json").write_text(json.dumps(
        {"kind": "text", "name": "dummy-text", "onnx": "model.onnx", "tokenizer": "tokenizer.json",
         "prefix": "query: ", "pooling": "mean", "max_length": 64}))
    return folder


def make_clip_model(folder: Path, dim: int = 8) -> Path:
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    folder.mkdir(parents=True, exist_ok=True)
    vocab = _tokenizer(WORDS, folder / "tokenizer.json", specials=("<|startoftext|>", "<|endoftext|>", "[UNK]"),
                       add_bos_eos=True)
    rng = np.random.default_rng(1)
    W = rng.normal(size=(3, dim)).astype(np.float32)
    E = rng.normal(size=(len(vocab), dim)).astype(np.float32)
    vision = helper.make_graph(
        [helper.make_node("ReduceMean", ["pixel_values"], ["pooled"], axes=[2, 3], keepdims=0),
         helper.make_node("MatMul", ["pooled", "W"], ["image_embeds"])],
        "vision",
        [helper.make_tensor_value_info("pixel_values", TensorProto.FLOAT, ["b", 3, 224, 224])],
        [helper.make_tensor_value_info("image_embeds", TensorProto.FLOAT, ["b", dim])],
        [numpy_helper.from_array(W, "W")],
    )
    text = helper.make_graph(
        [helper.make_node("Gather", ["E", "input_ids"], ["tok"], axis=0),
         helper.make_node("ReduceMean", ["tok"], ["text_embeds"], axes=[1], keepdims=0)],
        "textenc",
        [helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["b", 77])],
        [helper.make_tensor_value_info("text_embeds", TensorProto.FLOAT, ["b", dim])],
        [numpy_helper.from_array(E, "E")],
    )
    for g, name in ((vision, "vision_model.onnx"), (text, "text_model.onnx")):
        m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
        m.ir_version = 8
        onnx.save(m, str(folder / name))
    (folder / "filesorter_model.json").write_text(json.dumps(
        {"kind": "clip", "name": "dummy-clip", "vision_onnx": "vision_model.onnx", "text_onnx": "text_model.onnx",
         "tokenizer": "tokenizer.json", "image_size": 224}))
    return folder
