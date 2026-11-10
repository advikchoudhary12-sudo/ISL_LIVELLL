param(
    [ValidateSet('menu', 'train', 'webcam')]
    [string]$Action = 'menu',
    [string]$CameraIndex = ''
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$ScriptsDir = Join-Path $ProjectRoot '.venv\Scripts'

function Invoke-ProjectCommand {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [string[]]$Arguments = @()
    )

    $Executable = Join-Path $ScriptsDir ($Name + '.exe')
    if (-not (Test-Path -LiteralPath $Executable -PathType Leaf)) {
        throw "Project environment is missing $Name. Run .\windows\setup.ps1 first."
    }
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Name exited with code $LASTEXITCODE."
    }
}

if (-not (Test-Path -LiteralPath (Join-Path $ScriptsDir 'isl-webcam.exe') -PathType Leaf)) {
    throw 'The project environment is not set up. Run .\windows\setup.ps1 first.'
}

if ($Action -eq 'menu') {
    Write-Host 'ISL Sign Recognition'
    Write-Host '1) Prepare and train the model'
    Write-Host '2) Open webcam'
    $Choice = Read-Host 'Choose 1 or 2'
    switch ($Choice) {
        '1' { $Action = 'train' }
        '2' { $Action = 'webcam' }
        default { throw 'Choose 1 or 2.' }
    }
}

Push-Location $ProjectRoot
try {
    if ($Action -eq 'train') {
        Invoke-ProjectCommand -Name 'isl-prepare-static'
        Invoke-ProjectCommand -Name 'isl-train-static'
    } elseif ($Action -eq 'webcam') {
        if ([string]::IsNullOrWhiteSpace($CameraIndex)) {
            if (-not [string]::IsNullOrWhiteSpace($env:ISL_CAMERA)) {
                $CameraIndex = $env:ISL_CAMERA
            } elseif ($Action -eq 'webcam' -and $Choice -eq '2') {
                $CameraIndex = Read-Host 'Camera index (press Enter for 0)'
            }
        }
        if ([string]::IsNullOrWhiteSpace($CameraIndex)) { $CameraIndex = '0' }
        if ($CameraIndex -notmatch '^\d+$') {
            throw 'Camera index must be a non-negative integer.'
        }
        $ModelPath = Join-Path $ProjectRoot 'artifacts\static_model.pt'
        if (-not (Test-Path -LiteralPath $ModelPath -PathType Leaf)) {
            throw 'The trained model is missing. Run the training option first.'
        }
        Invoke-ProjectCommand -Name 'isl-webcam' -Arguments @('--camera', $CameraIndex)
    }
} finally {
    Pop-Location
}
