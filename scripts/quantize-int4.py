#!/usr/bin/env python
"""
Optional, further-shrinking step: quantizes an already-exported fp32 ONNX LLMLingua-2 model down
to INT4 weights using ONNX Runtime's own MatMulNBitsQuantizer -- both MatMul weights AND the
Gather-based embedding table. XLM-RoBERTa-large's word-embedding matrix (250002 x 1024) is ~1GB in
fp32 on its own; quantizing only MatMul (as an earlier version of this script did) leaves it
untouched, which is why that "INT4" output ended up larger than the INT8 community build. This is
a lossy, opt-in step, not run automatically by anything else in this project -- run
export-model.py first to produce the fp32 model this script quantizes; the fp32 source is never
modified.

Prerequisites (Python 3.10+, on top of export-model.py's onnxruntime/onnx):
    pip install onnx-ir "onnxruntime>=1.20.0"

Gather -> GatherBlockQuantized requires ONNX Runtime >= 1.20.0 to *execute* the result (both this
script's onnxruntime and the .NET CLI's Microsoft.ML.OnnxRuntime NuGet) -- this script checks that
at startup and refuses to silently fall back to a MatMul-only quantization on an older runtime.

Usage:
    python scripts/quantize-int4.py --input %LOCALAPPDATA%\\lingpack\\models\\llmlingua-2-xlm-roberta-large --output %LOCALAPPDATA%\\lingpack\\models\\llmlingua-2-xlm-roberta-large-int4
"""
from __future__ import annotations

import argparse
import pathlib
import shutil
import sys
from collections import Counter

import onnx

MIN_ONNXRUNTIME_VERSION = (1, 20, 0)

EXPECTED_FILES = ["model.onnx", "sentencepiece.bpe.model", "tokenizer.json", "config.json"]
# Only MatMul/Gather weight tensors change under quantization; everything else is copied through
# unchanged from the fp32 source directory.
COPIED_ASSETS = [
    "sentencepiece.bpe.model", "tokenizer.json", "config.json",
    "special_tokens_map.json", "tokenizer_config.json",
]

# Bytes per element by onnx.TensorProto.DataType, used only for the human-readable size reports
# below. INT4/UINT4 (onnx 1.16+) pack two values per byte; 0.5 here is an approximation.
_DTYPE_BYTES = {
    onnx.TensorProto.FLOAT: 4, onnx.TensorProto.UINT8: 1, onnx.TensorProto.INT8: 1,
    onnx.TensorProto.UINT16: 2, onnx.TensorProto.INT16: 2, onnx.TensorProto.INT32: 4,
    onnx.TensorProto.INT64: 8, onnx.TensorProto.BOOL: 1, onnx.TensorProto.FLOAT16: 2,
    onnx.TensorProto.DOUBLE: 8, onnx.TensorProto.UINT32: 4, onnx.TensorProto.UINT64: 8,
    onnx.TensorProto.BFLOAT16: 2, onnx.TensorProto.INT4: 0.5, onnx.TensorProto.UINT4: 0.5,
}


def dtype_name(data_type: int) -> str:
    return onnx.TensorProto.DataType.Name(data_type)


def initializer_size_bytes(init: onnx.TensorProto) -> float:
    n = 1
    for d in init.dims:
        n *= d
    return n * _DTYPE_BYTES.get(init.data_type, 4)


def print_largest_initializers(graph, title: str, top_n: int = 20, only_fp32: bool = False) -> None:
    inits = graph.initializer
    if only_fp32:
        inits = [i for i in inits if i.data_type == onnx.TensorProto.FLOAT]
    ranked = sorted(inits, key=initializer_size_bytes, reverse=True)[:top_n]
    print(f"\n{title} (top {len(ranked)}):")
    print(f"  {'size MB':>10}  {'dtype':<10} {'shape':<24} name")
    for init in ranked:
        size_mb = initializer_size_bytes(init) / (1024 * 1024)
        print(f"  {size_mb:>10.1f}  {dtype_name(init.data_type):<10} {str(list(init.dims)):<24} {init.name}")


def op_type_counts(graph) -> Counter:
    return Counter(n.op_type for n in graph.node)


def print_op_counts(counts: Counter, title: str) -> None:
    print(f"\n{title}:")
    for op, count in counts.most_common():
        print(f"  {op:<25} {count}")


def model_size_bytes(model_dir: pathlib.Path) -> int:
    total = 0
    # ".onnx.data" is what onnxruntime's quantizer writes (save_model_to_file); "_onnx_data" is
    # what optimum's exporter writes for the fp32 source -- check both naming conventions.
    for name in ("model.onnx", "model.onnx.data", "model.onnx_data"):
        p = model_dir / name
        if p.exists():
            total += p.stat().st_size
    return total


def check_onnxruntime_version() -> str:
    import onnxruntime
    version = onnxruntime.__version__
    parts = tuple(int(p) for p in version.split(".")[:3])
    if parts < MIN_ONNXRUNTIME_VERSION:
        min_str = ".".join(map(str, MIN_ONNXRUNTIME_VERSION))
        print(
            f"ERROR: onnxruntime {version} is too old for this script.\n"
            f"Quantizing the embedding table produces a GatherBlockQuantized node, which requires\n"
            f"ONNX Runtime >= {min_str} to *execute* -- both this script's own onnxruntime and the\n"
            f".NET CLI's Microsoft.ML.OnnxRuntime NuGet. Refusing to proceed rather than either\n"
            f"producing a model your runtime can't load, or silently falling back to a MatMul-only\n"
            f"quantization that leaves the ~1GB embedding table in fp32.\n"
            f"Upgrade with: pip install -U \"onnxruntime>={min_str}\"",
            file=sys.stderr,
        )
        sys.exit(1)
    return version


def find_embedding_initializer(graph):
    """The largest fp32 initializer feeding a Gather node -- i.e. the word-embedding table."""
    gather_inputs = {n.input[0] for n in graph.node if n.op_type == "Gather"}
    candidates = [i for i in graph.initializer if i.name in gather_inputs and i.data_type == onnx.TensorProto.FLOAT]
    if not candidates:
        return None
    return max(candidates, key=initializer_size_bytes)


def run_inference_smoke_test(in_dir: pathlib.Path, fp32_model: pathlib.Path, int4_model: pathlib.Path) -> None:
    import numpy as np
    import onnxruntime as ort
    from transformers import AutoTokenizer

    print("\nFunctional inference smoke test (fp32 vs INT4):")
    sample_text = "The quick brown fox jumps over the lazy dog while a gentle breeze moves through the trees."

    tokenizer = AutoTokenizer.from_pretrained(str(in_dir))
    encoded = tokenizer(sample_text, return_tensors="np")

    fp32_session = ort.InferenceSession(str(fp32_model), providers=["CPUExecutionProvider"])
    int4_session = ort.InferenceSession(str(int4_model), providers=["CPUExecutionProvider"])

    fp32_input_names = {i.name for i in fp32_session.get_inputs()}
    int4_input_names = {i.name for i in int4_session.get_inputs()}
    if fp32_input_names != int4_input_names:
        print(f"  WARNING: input name mismatch fp32={fp32_input_names} int4={int4_input_names}")
    else:
        print(f"  Input names match: {sorted(fp32_input_names)}")

    feed = {name: encoded[name] for name in fp32_input_names if name in encoded}

    fp32_logits = fp32_session.run(None, feed)[0]
    int4_logits = int4_session.run(None, feed)[0]

    print(f"  fp32 output shape: {fp32_logits.shape}, INT4 output shape: {int4_logits.shape}")
    if fp32_logits.shape != int4_logits.shape:
        print("  WARNING: output shapes differ -- skipping numeric comparison.")
        return

    print(f"  fp32 has NaN/Inf: {not np.isfinite(fp32_logits).all()}; INT4 has NaN/Inf: {not np.isfinite(int4_logits).all()}")

    diff = np.abs(fp32_logits - int4_logits)
    print(f"  logits mean absolute error: {diff.mean():.6f}")
    print(f"  logits max absolute error:  {diff.max():.6f}")

    fp32_keep = fp32_logits.argmax(axis=-1)
    int4_keep = int4_logits.argmax(axis=-1)
    agreement = (fp32_keep == int4_keep).mean()
    print(f"  KEEP/DROP token-level agreement: {agreement:.2%} (exact match not expected/required at 4-bit precision)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", "--in", dest="in_dir", required=True,
                         help="Directory with the fp32 ONNX export to quantize (from export-model.py).")
    parser.add_argument("--output", "--out", dest="out_dir", required=True,
                         help="Output directory for the INT4-quantized model + tokenizer assets.")
    parser.add_argument("--block-size", type=int, default=32,
                         help="Block size for weight-only quantization (default 32). Must be >=16 and a power of two.")
    sym_group = parser.add_mutually_exclusive_group()
    sym_group.add_argument(
        "--symmetric", dest="symmetric", action="store_true",
        help="Symmetric quantization (default): no zero-point tensor needed, matches ONNX "
             "Runtime's own default for RTN weight-only quantization.")
    sym_group.add_argument(
        "--asymmetric", dest="symmetric", action="store_false",
        help="Asymmetric quantization: adds a zero-point tensor per quantized op; can be "
             "marginally more accurate at a small extra storage cost.")
    parser.set_defaults(symmetric=True)
    parser.add_argument("--overwrite", action="store_true",
                         help="Allow overwriting an existing output model.onnx.")
    parser.add_argument("--verify", action=argparse.BooleanOptionalAction, default=True,
                         help="Run onnx.checker + InferenceSession load + fp32-vs-INT4 inference "
                              "smoke test after quantizing (default: on; use --no-verify to skip).")
    args = parser.parse_args()

    if args.block_size < 16 or (args.block_size & (args.block_size - 1)) != 0:
        print(f"ERROR: --block-size must be >= 16 and a power of two (got {args.block_size}).", file=sys.stderr)
        sys.exit(1)

    ort_version = check_onnxruntime_version()

    in_dir = pathlib.Path(args.in_dir).expanduser().resolve()
    out_dir = pathlib.Path(args.out_dir).expanduser().resolve()

    if in_dir == out_dir:
        print("ERROR: --input and --output must be different directories (the fp32 source is never modified).", file=sys.stderr)
        sys.exit(1)

    in_model = in_dir / "model.onnx"
    if not in_model.exists():
        print(f"ERROR: {in_model} not found. Run scripts/export-model.py first to produce the fp32 model.", file=sys.stderr)
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_model = out_dir / "model.onnx"
    if out_model.exists() and not args.overwrite:
        print(f"ERROR: {out_model} already exists. Pass --overwrite to replace it.", file=sys.stderr)
        sys.exit(1)
    if out_model.exists():
        out_model.unlink()
        for stale in (out_dir / "model.onnx.data", out_dir / "model.onnx_data"):
            if stale.exists():
                stale.unlink()

    print("=" * 70)
    print(f"Source model:         {in_model}")
    print(f"Destination:          {out_model}")
    print(f"onnxruntime:          {ort_version}")
    print(f"bits:                 4")
    print(f"block_size:           {args.block_size}")
    print(f"symmetric:            {args.symmetric}")
    print(f"op_types_to_quantize: MatMul, Gather")
    print("=" * 70)

    source_bytes = model_size_bytes(in_dir)
    print(f"\nSource model size: {source_bytes / (1024 ** 3):.2f} GB")

    before_model = onnx.load(str(in_model), load_external_data=False)
    before_counts = op_type_counts(before_model.graph)
    print_op_counts(before_counts, "Operator counts BEFORE quantization")
    print(f"  (MatMul: {before_counts.get('MatMul', 0)}, Gather: {before_counts.get('Gather', 0)})")
    print_largest_initializers(before_model.graph, "Largest initializers BEFORE quantization")

    embedding_before = find_embedding_initializer(before_model.graph)
    embedding_name = embedding_before.name if embedding_before is not None else None
    if embedding_before is not None:
        print(f"\nIdentified embedding initializer to track: '{embedding_name}' "
              f"({initializer_size_bytes(embedding_before) / (1024 ** 2):.1f} MB fp32)")

    # Imported here (not at module top) so --help / arg-validation errors don't require onnx-ir to
    # be installed first.
    from onnxruntime.quantization.matmul_nbits_quantizer import (
        DefaultWeightOnlyQuantConfig, MatMulNBitsQuantizer, QuantFormat,
    )

    quant_config = DefaultWeightOnlyQuantConfig(
        block_size=args.block_size,
        is_symmetric=args.symmetric,
        # Gather quantization only supports QOperator format (matmul_nbits_quantizer.py asserts
        # this explicitly) -- QDQ is not an option here.
        quant_format=QuantFormat.QOperator,
        # The actual fix over the old MatMul-only script: also quantize Gather, which is how the
        # ~1GB word-embedding table is represented in this model.
        op_types_to_quantize=("MatMul", "Gather"),
        # ORT's own defaults: MatMul quantizes along axis 0 (output-feature blocks), Gather along
        # axis 1 (per-block scales along the hidden/embedding dimension of each row).
        quant_axes=(("MatMul", 0), ("Gather", 1)),
        bits=4,  # Gather quantization only supports 4 bits, per the library's own validation.
    )

    print("\nQuantizing (the embedding table is the slow part; this can take a few minutes)...")
    quant = MatMulNBitsQuantizer(model=str(in_model), bits=4, algo_config=quant_config)
    quant.process()
    quant.model.save_model_to_file(str(out_model), True)

    after_graph = quant.model.model.graph
    after_counts = op_type_counts(after_graph)
    print_op_counts(after_counts, "Operator counts AFTER quantization")
    print(f"  (MatMulNBits: {after_counts.get('MatMulNBits', 0)}, "
          f"GatherBlockQuantized: {after_counts.get('GatherBlockQuantized', 0)}, "
          f"remaining MatMul: {after_counts.get('MatMul', 0)}, "
          f"remaining Gather: {after_counts.get('Gather', 0)})")

    # Hard guard against exactly the bug this rewrite fixes: never report success if the
    # embedding table is still sitting in the output graph as a large fp32 tensor.
    if embedding_name is not None:
        still_fp32 = next(
            (i for i in after_graph.initializer if i.name == embedding_name and i.data_type == onnx.TensorProto.FLOAT),
            None,
        )
        if still_fp32 is not None and initializer_size_bytes(still_fp32) > 50 * 1024 * 1024:
            print(
                f"\nERROR: embedding initializer '{embedding_name}' is still fp32 "
                f"({initializer_size_bytes(still_fp32) / (1024 ** 2):.1f} MB) after quantization -- "
                "Gather quantization did not apply. Not reporting success; investigate before "
                "relying on this output.",
                file=sys.stderr,
            )
            sys.exit(1)

    for asset in COPIED_ASSETS:
        src = in_dir / asset
        if src.exists():
            shutil.copyfile(src, out_dir / asset)

    missing = [f for f in EXPECTED_FILES if not (out_dir / f).exists()]
    if missing:
        print(f"ERROR: expected output files missing: {missing}", file=sys.stderr)
        sys.exit(1)

    dest_bytes = model_size_bytes(out_dir)
    ratio = source_bytes / dest_bytes if dest_bytes else float("inf")
    print(f"\nDestination model size: {dest_bytes / (1024 ** 3):.2f} GB")
    print(f"Compression ratio (fp32 source / this output): {ratio:.2f}x")

    if args.verify:
        print("\nVerifying...")
        try:
            onnx.checker.check_model(str(out_model), full_check=True)
            print("  onnx.checker: passed (full_check)")
        except Exception as e:
            # com.microsoft contrib ops (MatMulNBits/GatherBlockQuantized) may not be fully
            # understood by the plain checker -- that's not evidence the model is actually broken.
            print(f"  onnx.checker: skipped/failed on custom ops ({type(e).__name__}: {e})")

        import onnxruntime as ort
        try:
            ort.InferenceSession(str(out_model), providers=["CPUExecutionProvider"])
            print("  onnxruntime.InferenceSession: loaded successfully")
        except Exception as e:
            print(f"ERROR: onnxruntime failed to load the quantized model: {e}", file=sys.stderr)
            sys.exit(1)

        if dest_bytes > 700 * 1024 * 1024:
            print(f"\nOutput is still > 700 MB ({dest_bytes / (1024 ** 2):.0f} MB) -- largest remaining fp32 tensors:")
            print_largest_initializers(after_graph, "Largest FP32 initializers remaining AFTER quantization", only_fp32=True)

        run_inference_smoke_test(in_dir, in_model, out_model)

    print(f"\nDone. INT4 model assets are at: {out_dir}")
    print("Point the CLI at it with --model-dir, or set LINGPACK_MODEL_DIR.")
    print("Note: this is a lossy, more aggressive quantization than fp32/INT8 -- see README for tradeoffs.")


if __name__ == "__main__":
    main()
