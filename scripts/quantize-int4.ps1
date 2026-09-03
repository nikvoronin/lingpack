<#
.SYNOPSIS
    Optional, further-shrinking step: quantizes the fp32 ONNX model (from export-model.ps1) down
    to INT4 weights via ONNX Runtime's own MatMulNBitsQuantizer -- both MatMul weights and the
    Gather-based embedding table -- for a smaller/faster model at the cost of more accuracy loss.
    Opt-in only -- nothing else in this project runs this automatically.

.PARAMETER InDir
    Directory with the fp32 ONNX export to quantize. Defaults to export-model.ps1's default output
    location.

.PARAMETER OutDir
    Output directory for the INT4-quantized model + tokenizer assets. Defaults to a directory
    distinct from the fp32/INT8-community ones, so this never silently overwrites either.

.PARAMETER BlockSize
    Block size for weight-only quantization (default 32). Must be >=16 and a power of two.

.PARAMETER Symmetric
    Use symmetric quantization (default $true) -- matches ONNX Runtime's own RTN default and needs
    no zero-point tensor. Pass -Symmetric:$false for asymmetric quantization instead.

.PARAMETER Overwrite
    Allow overwriting an existing output model.onnx.

.PARAMETER Verify
    Run onnx.checker + InferenceSession load + fp32-vs-INT4 inference smoke test after quantizing
    (default $true).

.PARAMETER PythonExe
    Path (or PATH-resolvable name) of the Python 3.10+ interpreter to use. Defaults to "python".

.EXAMPLE
    .\scripts\quantize-int4.ps1

.EXAMPLE
    .\scripts\quantize-int4.ps1 -Overwrite -Symmetric:$false
#>
param(
    [string]$InDir = (Join-Path $env:LOCALAPPDATA "lingpack\models\llmlingua-2-xlm-roberta-large"),
    [string]$OutDir = (Join-Path $env:LOCALAPPDATA "lingpack\models\llmlingua-2-xlm-roberta-large-int4"),
    [int]$BlockSize = 32,
    [bool]$Symmetric = $true,
    [switch]$Overwrite,
    [bool]$Verify = $true,
    [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$venvDir = Join-Path $repoRoot ".export-venv"

if (-not (Test-Path $venvDir)) {
    Write-Host "Creating Python venv at $venvDir"
    & $PythonExe -m venv $venvDir
}

$venvPython = Join-Path $venvDir "Scripts\python.exe"

# Gather -> GatherBlockQuantized quantization requires ONNX Runtime >= 1.20.0 to execute; pin that
# floor here as defense in depth alongside quantize-int4.py's own runtime version check.
Write-Host "Installing/upgrading quantization dependencies (onnx-ir, onnxruntime>=1.20.0)..."
& $venvPython -m pip install onnx-ir "onnxruntime>=1.20.0"

$pyArgs = @(
    (Join-Path $PSScriptRoot "quantize-int4.py"),
    "--input", $InDir,
    "--output", $OutDir,
    "--block-size", $BlockSize
)
$pyArgs += if ($Symmetric) { "--symmetric" } else { "--asymmetric" }
if ($Overwrite) { $pyArgs += "--overwrite" }
$pyArgs += if ($Verify) { "--verify" } else { "--no-verify" }

Write-Host "Quantizing model to INT4 (MatMul + Gather): $InDir -> $OutDir"
& $venvPython @pyArgs

Write-Host "Done. INT4 model assets are at: $OutDir"
