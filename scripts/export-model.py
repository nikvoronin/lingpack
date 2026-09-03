#!/usr/bin/env python
"""
One-time helper: exports microsoft/llmlingua-2-xlm-roberta-large-meetingbank to ONNX for use by
the lingpack .NET CLI. This is run once, by hand, with Python -- the .NET app itself never runs
Python or does any model conversion.

Prerequisites (Python 3.10+):
    pip install optimum optimum-onnx onnx onnxruntime transformers

Usage:
    python scripts/export-model.py --out %LOCALAPPDATA%\\lingpack\\models\\llmlingua-2-xlm-roberta-large
"""
import argparse
import pathlib
import shutil
import subprocess
import sys

DEFAULT_MODEL_ID = "microsoft/llmlingua-2-xlm-roberta-large-meetingbank"
EXPECTED_FILES = ["model.onnx", "sentencepiece.bpe.model", "tokenizer.json", "config.json"]

# microsoft/llmlingua-2-xlm-roberta-large-meetingbank fine-tunes FacebookAI/xlm-roberta-large's
# token-classification head but its HF repo only ships the "fast" tokenizer.json, not the raw
# SentencePiece proto -- LingPack.Core's tokenizer needs that proto for Unigram segmentation. The
# fine-tune reuses the base model's tokenizer unchanged, so we fetch it from the base repo instead.
SENTENCEPIECE_SOURCE_MODEL_ID = "FacebookAI/xlm-roberta-large"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="Hugging Face model id to export.")
    parser.add_argument("--out", required=True, help="Output directory for the exported ONNX model + tokenizer assets.")
    parser.add_argument("--opset", type=int, default=None, help="Override the ONNX opset (defaults to optimum's choice).")
    parser.add_argument(
        "--sentencepiece-source", default=SENTENCEPIECE_SOURCE_MODEL_ID,
        help="HF model id to fetch sentencepiece.bpe.model from, if --model-id's repo doesn't ship it.")
    args = parser.parse_args()

    out_dir = pathlib.Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "optimum.exporters.onnx",
        "--model", args.model_id,
        "--task", "token-classification",
        str(out_dir),
    ]
    if args.opset:
        cmd += ["--opset", str(args.opset)]

    print(f"Exporting {args.model_id} -> {out_dir}")
    subprocess.run(cmd, check=True)

    if not (out_dir / "sentencepiece.bpe.model").exists():
        print(f"{args.model_id} doesn't ship sentencepiece.bpe.model; "
              f"fetching it from {args.sentencepiece_source} instead (same tokenizer/vocab).")
        from huggingface_hub import hf_hub_download
        fetched = hf_hub_download(repo_id=args.sentencepiece_source, filename="sentencepiece.bpe.model")
        shutil.copyfile(fetched, out_dir / "sentencepiece.bpe.model")

    missing = [f for f in EXPECTED_FILES if not (out_dir / f).exists()]
    if missing:
        print(f"WARNING: expected export files missing: {missing}", file=sys.stderr)
        sys.exit(1)

    print(f"Export OK -> {out_dir}")
    print("Point the CLI at it with --model-dir, or set LINGPACK_MODEL_DIR, "
          "or leave it at the default location shown above.")


if __name__ == "__main__":
    main()
