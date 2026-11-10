$ErrorActionPreference = 'Stop'

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VenvPath = Join-Path $ProjectRoot '.venv'
$PythonPath = Join-Path $VenvPath 'Scripts\python.exe'

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw 'uv was not found. Install uv, then run this setup script again.'
}

Push-Location $ProjectRoot
try {
    & uv venv --python 3.12 $VenvPath
    if ($LASTEXITCODE -ne 0) { throw "uv venv failed with exit code $LASTEXITCODE" }

    & uv pip install --python $PythonPath -r (Join-Path $ProjectRoot 'requirements.txt')
    if ($LASTEXITCODE -ne 0) { throw "Installing application dependencies failed with exit code $LASTEXITCODE" }

    & uv pip install --python $PythonPath -r (Join-Path $ProjectRoot 'requirements-torch-cpu.txt')
    if ($LASTEXITCODE -ne 0) { throw "Installing CPU PyTorch failed with exit code $LASTEXITCODE" }

    & uv pip install --python $PythonPath --no-deps -e $ProjectRoot
    if ($LASTEXITCODE -ne 0) { throw "Installing the ISL application failed with exit code $LASTEXITCODE" }
} finally {
    Pop-Location
}

Write-Host ''
Write-Host 'Windows setup is complete.' -ForegroundColor Green
Write-Host 'Double-click "Run ISL on Windows.bat" or run .\windows\isl-launcher.ps1.'
