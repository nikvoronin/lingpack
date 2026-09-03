<#
.SYNOPSIS
    One-time helper: sets up a local Python venv and exports
    microsoft/llmlingua-2-xlm-roberta-large-meetingbank to ONNX for use by the lingpack .NET CLI.

.PARAMETER OutDir
    Output directory for the exported ONNX model + tokenizer assets.
    Defaults to %LOCALAPPDATA%\lingpack\models\llmlingua-2-xlm-roberta-large (lingpack's own default
    --model-dir location).

.PARAMETER PythonExe
    Path (or PATH-resolvable name) of the Python 3.10+ interpreter to use for creating the venv.
    Defaults to "python"; override this if Python isn't on PATH in the current session (e.g. right
    after installing it, before PATH has been refreshed).

.EXAMPLE
    .\scripts\export-model.ps1

.EXAMPLE
    .\scripts\export-model.ps1 -PythonExe "C:\Users\me\AppData\Local\Programs\Python\Python312\python.exe"
#>
param(
    [string]$OutDir = (Join-Path $env:LOCALAPPDATA "lingpack\models\llmlingua-2-xlm-roberta-large"),
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

Write-Host "Installing export dependencies (optimum, optimum-onnx, onnx, onnxruntime, transformers)..."
& $venvPython -m pip install --upgrade pip | Out-Null
# optimum's ONNX exporter now ships as the separate "optimum-onnx" package (the old
# `optimum[exporters]` extra no longer provides it as of optimum 2.x).
& $venvPython -m pip install optimum optimum-onnx onnx onnxruntime transformers

Write-Host "Exporting model to $OutDir"
& $venvPython (Join-Path $PSScriptRoot "export-model.py") --out $OutDir

Write-Host "Done. Model assets are at: $OutDir"
