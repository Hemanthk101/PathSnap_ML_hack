param(
    [ValidateRange(1, 65535)] [int] $Port = 8765,
    [switch] $Lan
)

$ErrorActionPreference = 'Stop'
$appRoot = $PSScriptRoot
$python = Join-Path $appRoot '.venv/Scripts/python.exe'
if (!(Test-Path -LiteralPath $python)) {
    $python = Join-Path (Split-Path $appRoot -Parent) '.venv/Scripts/python.exe'
}
if (!(Test-Path -LiteralPath $python)) {
    throw 'Create .venv and install requirements.txt from the PathSnap project directory.'
}
$binding = if ($Lan) { '0.0.0.0' } else { '127.0.0.1' }
$localEvaluator = Join-Path $appRoot 'data/evaluator/trips'
$localReferences = Join-Path $appRoot 'data/evaluator/references'
$importedEvaluator = !$env:PATHSNAP_EVALUATOR_ROOT -and (Test-Path -LiteralPath $localEvaluator)
$setLocalReferences = $importedEvaluator -and !$env:PATHSNAP_EVALUATOR_REFERENCE_ROOT -and (Test-Path -LiteralPath $localReferences)
if ($importedEvaluator) { $env:PATHSNAP_EVALUATOR_ROOT = $localEvaluator }
if ($setLocalReferences) { $env:PATHSNAP_EVALUATOR_REFERENCE_ROOT = $localReferences }
Push-Location -LiteralPath $appRoot
try {
    & $python -m uvicorn pathsnap.api:app --host $binding --port $Port
    exit $LASTEXITCODE
} finally {
    Pop-Location
    if ($importedEvaluator) { Remove-Item -LiteralPath Env:PATHSNAP_EVALUATOR_ROOT }
    if ($setLocalReferences) { Remove-Item -LiteralPath Env:PATHSNAP_EVALUATOR_REFERENCE_ROOT }
}
