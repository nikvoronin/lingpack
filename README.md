# lingpack

A .NET 10 console utility that compresses prompts using Microsoft's [LLMLingua-2](https://www.microsoft.com/en-us/research/project/llmlingua/llmlingua-2/) algorithm: a
small fine-tuned token-classification model scores every word in a prompt by how essential it is,
and the lowest-scoring words are dropped to hit a target compression rate or token budget.

- [Model setup (one-time)](#model-setup-one-time)
    - [Alternative: download a pre-converted ONNX model (no Python)](#alternative-download-a-pre-converted-onnx-model-no-python)
    - [Troubleshooting: tokenizer id mapping](#troubleshooting-tokenizer-id-mapping)
- [Usage](#usage)
- [Building and testing](#building-and-testing)
- [Further shrinking the model](#further-shrinking-the-model)
    - [INT4 weight quantization (opt-in)](#int4-weight-quantization-opt-in)

## Model setup (one-time)

The compressor needs the `microsoft/llmlingua-2-xlm-roberta-large-meetingbank` model exported to ONNX. This is
a one-time step done with Python — the .NET app itself never runs Python or converts anything.

1. Prerequisites: Python 3.10+.

    ```shell
    pip install optimum optimum-onnx onnx onnxruntime transformers
    ```

2. Run the export script:

     ```shell
     python scripts/export-model.py --out %LOCALAPPDATA%\lingpack\models\llmlingua-2-xlm-roberta-large
     ```

   or, on Windows, the convenience wrapper that also sets up an isolated venv:

    ```shell
    .\scripts\export-model.ps1
    ```

3. Confirm the output directory contains `model.onnx`, `sentencepiece.bpe.model`, `tokenizer.json`, and `config.json`.

The CLI resolves the model directory in this order: `--model-dir` flag > `LINGPACK_MODEL_DIR`
environment variable > `%LOCALAPPDATA%\lingpack\models\llmlingua-2-xlm-roberta-large`.

### Alternative: download a pre-converted ONNX model (no Python)

Converting the original PyTorch checkpoint to ONNX inherently requires tracing a PyTorch graph —
there's no way around Python for that conversion step itself. But if you'd rather not install
Python at all, you can instead download an ONNX build someone else already converted:

```shell
.\scripts\download-model.ps1
```

This pulls a **third-party, INT8-quantized** conversion of the model (not published by Microsoft)
straight from Hugging Face Hub over plain HTTPS — no Python, PyTorch, or `optimum` involved. It's
a real trade-off, not a free lunch: quantization and unverified provenance mean its output may
differ slightly from the full-precision export above, which is the one this project actually
verified end-to-end. Prefer `export-model.ps1` when Python is available; reach for this only when
it isn't. It writes to a different default directory than `export-model.ps1` so it never silently
overwrites an existing export — point the CLI at it explicitly:

```shell
dotnet run --project src/LingPack.Cli -- compress --rate 0.5 --model-dir %LOCALAPPDATA%\lingpack\models\llmlingua-2-xlm-roberta-large-int8-community
```

### Troubleshooting: tokenizer id mapping

XLM-RoBERTa remaps raw SentencePiece piece ids to different final vocabulary ids (a "fairseq
offset": ordinary pieces are `sp_id + 1`, specials are fixed at `0-3`, `<mask>` is appended at the
end). `LingPack.Core`'s `XlmRobertaTokenizerAdapter` avoids re-deriving that formula: it uses the
SentencePiece model only for segmentation, then resolves each resulting piece to its authoritative
id via the `tokenizer.json` exported alongside the model. If compressed output ever looks
nonsensical (e.g. compression seems to drop essential words at random), verify first that
`tokenizer.json` exists next to `model.onnx` and matches the same export — that mapping is the
most likely culprit.

## Usage

```shell
dotnet run --project src/LingPack.Cli -- compress --input prompt.txt --rate 0.5
```

- `--input <file>` — omit to read from stdin.
- `--output <file>` — omit to write the compressed prompt to stdout.
- `--rate <0..1>` or `--target-tokens <N>` — exactly one is required.
- `--model-dir <path>` — override the model directory (see above).
- `--force-token <word>` — repeatable; a word that is always kept regardless of its score.
- `--chunk-max-tokens <N>` — model max sequence length used when chunking long inputs (default 512).
- `--verbose` — print per-word scores and compression stats to stderr.

Example:

```shell
echo "The quick brown fox jumps over the lazy dog." | dotnet run --project src/LingPack.Cli -- compress --rate 0.5 --verbose
```

## Building and testing

```shell
dotnet build lingpack.slnx
dotnet test tests/LingPack.Core.Tests
```

The unit test suite covers word splitting, chunking, and the ranking/reassembly logic entirely
with fakes, so it passes without the real model. A separate smoke test suite
(`OnnxTokenClassifierSmokeTests`) exercises real inference once the model has been exported;
run it explicitly with:

```shell
dotnet test tests/LingPack.Core.Tests --filter Category=RequiresModel
```

## Further shrinking the model

### INT4 weight quantization (opt-in)

If you already have the fp32 export from `export-model.ps1` and want an even smaller/faster model,
you can quantize its weights down to INT4 — both `MatMul` weights **and** the embedding table —
via ONNX Runtime's own `onnxruntime.quantization.matmul_nbits_quantizer`:

```shell
pip install onnx-ir "onnxruntime>=1.20.0"
.\scripts\quantize-int4.ps1 -Overwrite
```

This is a **more aggressive, lossy** step than the INT8 community build above — opt-in only,
nothing else in this project runs it automatically.

**MatMul-only vs. full (MatMul+Gather) INT4.** XLM-RoBERTa-large's word-embedding table
(`250002 × 1024`) is implemented as a `Gather` lookup, not a `MatMul`, and it alone is ~1&nbsp;GB in
fp32 — nearly half the model. An earlier version of this script only quantized `MatMul`:

```text
embedding (Gather) -> stays FP32 (~1 GB)
MatMul             ->> INT4

total: fp32 2.2 GB -> ~1.2 GB     (bigger than the 560 MB INT8 build!)
```

`onnxruntime.quantization.matmul_nbits_quantizer` also supports quantizing `Gather` directly
(`op_types_to_quantize=("MatMul", "Gather")`), producing a `GatherBlockQuantized` op alongside
`MatMulNBits` — the current script uses this:

```text
embedding (Gather) -> GatherBlockQuantized (INT4)
MatMul             ->> MatMulNBits (INT4)

total: fp32 2.2 GB -> ~0.33 GB     (smaller than the 560 MB INT8 build)
```

This was verified directly (not just inferred from file size) by inspecting the INT8 community
build's own graph: its `word_embeddings` `Gather` node indexes into an actual `UINT8`-dtype
initializer, confirming that pipeline quantizes the embedding too — which is exactly what the old
MatMul-only INT4 script was missing, and why it came out larger than the INT8 build despite using
fewer bits per weight.

**Compatibility.** Executing a `GatherBlockQuantized` node requires **ONNX Runtime ≥ 1.20.0** —
both the Python quantization tooling and the .NET CLI's `Microsoft.ML.OnnxRuntime` NuGet (already
pinned at `1.29.0` in `src/LingPack.Core/LingPack.Core.csproj`, well above the floor). `scripts/quantize-int4.py`
checks its own `onnxruntime.__version__` at startup and refuses to run on anything older — it will
not silently fall back to MatMul-only quantization, since that reproduces the exact "bigger than
INT8" problem above without telling you why.

**Verified end-to-end in this project**, quantizing the actual exported fp32 model:

- Operator counts: `MatMulNBits: 145`, `GatherBlockQuantized: 3` (the token/position/type embedding
  tables), 48 `MatMul` and 52 `Gather` nodes deliberately left unquantized (non-constant-weight ops,
  e.g. attention score matmuls and position-id slicing — quantizing those isn't meaningful).
- Size: **2.08 GB (fp32) → 0.33 GB (INT4)**, a 6.4x reduction — smaller than the 560 MB INT8 build.
- `onnx.checker.check_model(full_check=True)` passed; `onnxruntime.InferenceSession` loads the
  result; the .NET CLI runs it successfully with the same qualitative output as fp32/INT8.
- fp32-vs-INT4 inference smoke test on a sample sentence: matching input names and output shapes,
  no NaN/Inf in either output, logits mean-absolute-error ≈0.33 / max-absolute-error ≈1.16, and
  **100% token-level KEEP/DROP agreement** with the fp32 model on that sentence (exact numeric
  match isn't expected or required at 4-bit precision — the decision agreement is what matters).

Point the CLI at the result the same way:

```shell
dotnet run --project src/LingPack.Cli -- compress --rate 0.5 --model-dir %LOCALAPPDATA%\lingpack\models\llmlingua-2-xlm-roberta-large-int4
```

Useful flags: `-BlockSize` (default 32), `-Symmetric:$false` for asymmetric quantization,
`-Overwrite` to replace an existing output, `-Verify:$false` to skip the checker/inference checks
above (on by default).
