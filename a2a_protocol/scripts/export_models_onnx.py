#!/usr/bin/env python3
"""
One-time script: Export ML models to ONNX + int8 quantization.

Run this ONCE locally before building the Docker image.
The output goes into a2a_protocol/models/ and gets baked into the image.

Local requirements (NOT needed in Docker):
    pip install optimum[onnxruntime] onnxruntime torch sentence-transformers

What this does:
    1. Downloads both models from HuggingFace
    2. Converts them from PyTorch → ONNX format
    3. Applies dynamic int8 quantization (4x size reduction on model weights)
    4. Saves tokenizer files alongside each model
    5. Deletes the large FP32 .onnx files — only keeps the int8 quantized ones

Output structure:
    models/
    ├── embedding/
    │   ├── model_quantized.onnx   (~22 MB, was 90 MB)
    │   ├── tokenizer.json
    │   ├── tokenizer_config.json
    │   ├── vocab.txt
    │   └── special_tokens_map.json
    └── reranker/
        ├── model_quantized.onnx   (~22 MB, was 90 MB)
        ├── tokenizer.json
        ├── tokenizer_config.json
        ├── vocab.txt
        └── special_tokens_map.json
"""

import os
import sys

from onnxruntime.quantization import QuantType, quantize_dynamic
from optimum.onnxruntime import (
    ORTModelForFeatureExtraction,
    ORTModelForSequenceClassification,
)
from transformers import AutoTokenizer

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")


def export_embedding():
    out = os.path.join(MODELS_DIR, "embedding")
    print(f"\n[1/2] Exporting embedding model → {out}")

    model = ORTModelForFeatureExtraction.from_pretrained(
        "sentence-transformers/all-MiniLM-L6-v2",
        export=True,
    )
    model.save_pretrained(out)

    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    tokenizer.save_pretrained(out)

    onnx_path = os.path.join(out, "model.onnx")
    quantized_path = os.path.join(out, "model_quantized.onnx")

    print("  Quantizing to int8 ...")
    quantize_dynamic(onnx_path, quantized_path, weight_type=QuantType.QInt8)

    # Delete the large FP32 model — only keep int8
    os.remove(onnx_path)
    size_mb = os.path.getsize(quantized_path) / 1024 / 1024
    print(f"  Done. model_quantized.onnx = {size_mb:.1f} MB")


def export_reranker():
    out = os.path.join(MODELS_DIR, "reranker")
    print(f"\n[2/2] Exporting reranker model → {out}")

    model = ORTModelForSequenceClassification.from_pretrained(
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
        export=True,
    )
    model.save_pretrained(out)

    tokenizer = AutoTokenizer.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2")
    tokenizer.save_pretrained(out)

    onnx_path = os.path.join(out, "model.onnx")
    quantized_path = os.path.join(out, "model_quantized.onnx")

    print("  Quantizing to int8 ...")
    quantize_dynamic(onnx_path, quantized_path, weight_type=QuantType.QInt8)

    os.remove(onnx_path)
    size_mb = os.path.getsize(quantized_path) / 1024 / 1024
    print(f"  Done. model_quantized.onnx = {size_mb:.1f} MB")


if __name__ == "__main__":
    os.makedirs(MODELS_DIR, exist_ok=True)

    try:
        export_embedding()
        export_reranker()
    except ImportError as e:
        print(f"\nMissing dependency: {e}")
        print("Install with: pip install optimum[onnxruntime] onnxruntime torch sentence-transformers")
        sys.exit(1)

    total_mb = sum(
        os.path.getsize(os.path.join(root, f)) / 1024 / 1024
        for root, _, files in os.walk(MODELS_DIR)
        for f in files
    )
    print(f"\nAll done! Total models/ size: {total_mb:.1f} MB")
    print("You can now build the Docker image.")
