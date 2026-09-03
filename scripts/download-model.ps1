<#
.SYNOPSIS
    Python-free alternative to export-model.ps1: downloads an already-converted ONNX build of the
    LLMLingua-2 model directly from Hugging Face Hub over plain HTTPS, for use by the lingpack
    .NET CLI. No Python, PyTorch, or optimum required.

.DESCRIPTION
    export-model.ps1 (Python-based, full precision) is the recommended default -- it exports
    directly from Microsoft's original weights and is what this project verified end-to-end.

    This script instead downloads a third-party, INT8-quantized ONNX conversion of the same model
    published by a community member (not Microsoft). Precision/behavior may differ slightly from
    the full-precision export due to quantization, and provenance isn't Microsoft-verified. Use
    this only when installing Python isn't an option.

    Like the official export, the source repo doesn't ship the raw SentencePiece proto
    (sentencepiece.bpe.model) -- it's fetched from the base FacebookAI/xlm-roberta-large repo
    instead, since the fine-tune reuses that tokenizer/vocab unchanged.

.PARAMETER OutDir
    Output directory for the downloaded ONNX model + tokenizer assets. Defaults to a directory
    distinct from export-model.ps1's default, so running this script never silently overwrites an
    existing full-precision export -- point the CLI at it explicitly via --model-dir to try it.

.PARAMETER OnnxSourceRepo
    Hugging Face repo id to download the pre-converted ONNX model + tokenizer files from.

.PARAMETER SentencePieceSourceRepo
    Hugging Face repo id to download sentencepiece.bpe.model from.

.EXAMPLE
    .\scripts\download-model.ps1

.EXAMPLE
    .\scripts\download-model.ps1 -OutDir "C:\models\llmlingua2-int8"
#>
param(
    [string]$OutDir = (Join-Path $env:LOCALAPPDATA "lingpack\models\llmlingua-2-xlm-roberta-large-int8-community"),
    [string]$OnnxSourceRepo = "KatawaDead/llmlingua-2-xlm-roberta-large-meetingbank-onnx-int8",
    [string]$SentencePieceSourceRepo = "FacebookAI/xlm-roberta-large"
)

$ErrorActionPreference = "Stop"

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

function Get-HuggingFaceFile {
    param([string]$RepoId, [string]$FileName, [string]$OutDir)

    $uri = "https://huggingface.co/$RepoId/resolve/main/$FileName"
    $outFile = Join-Path $OutDir $FileName
    Write-Host "Downloading $uri"
    Invoke-WebRequest -Uri $uri -OutFile $outFile
}

$onnxFiles = @("config.json", "model.onnx", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json")
foreach ($file in $onnxFiles) {
    Get-HuggingFaceFile -RepoId $OnnxSourceRepo -FileName $file -OutDir $OutDir
}

Get-HuggingFaceFile -RepoId $SentencePieceSourceRepo -FileName "sentencepiece.bpe.model" -OutDir $OutDir

$expectedFiles = @("model.onnx", "sentencepiece.bpe.model", "tokenizer.json", "config.json")
$missing = $expectedFiles | Where-Object { -not (Test-Path (Join-Path $OutDir $_)) }
if ($missing) {
    Write-Error "Missing expected file(s) after download: $($missing -join ', ')"
    exit 1
}

Write-Host "Done. Model assets are at: $OutDir"
Write-Host "Point the CLI at it with --model-dir `"$OutDir`", or set LINGPACK_MODEL_DIR."
Write-Host "Note: this is a third-party, INT8-quantized conversion (not from Microsoft) -- see README.md for the trade-offs vs. export-model.ps1."
