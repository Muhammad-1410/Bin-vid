<#
.SYNOPSIS
    Builds a standalone executable/installer distribution for binvid using PyInstaller.

.DESCRIPTION
    Packages binvid into dist/binvid/ (onedir, default) or dist/binvid.exe (onefile).
    Automatically bundles ffmpeg.exe and consola.ttf so the distribution is 100% self-contained.

.PARAMETER Mode
    Build mode: "onedir" (recommended for fast launch) or "onefile". Default: "onedir".

.PARAMETER Clean
    If specified, removes existing build/ and dist/ folders before building.
#>
param (
    [ValidateSet("onedir", "onefile")]
    [string]$Mode = "onedir",

    [switch]$Clean
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $ScriptDir

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "   binvid PyInstaller Build Pipeline" -ForegroundColor Cyan
Write-Host "   Mode: $Mode" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan

# 1. Locate Python and PyInstaller
$PythonExe = Join-Path $ScriptDir "venv\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
    $PythonExe = (Get-Command python -ErrorAction SilentlyContinue).Source
    if (-not $PythonExe) {
        Write-Error "Python executable not found. Please activate your virtual environment or install Python."
    }
}
Write-Host "[+] Using Python: $PythonExe" -ForegroundColor Green

# 2. Check bundled binaries: ffmpeg.exe
$BinDir = Join-Path $ScriptDir "bin"
$FfmpegDest = Join-Path $BinDir "ffmpeg.exe"
if (-not (Test-Path $FfmpegDest)) {
    Write-Host "[*] bin/ffmpeg.exe not found. Attempting to locate on system..." -ForegroundColor Yellow
    $SystemFfmpeg = (Get-Command ffmpeg -ErrorAction SilentlyContinue).Source
    if ($SystemFfmpeg) {
        New-Item -ItemType Directory -Force -Path $BinDir | Out-Null
        Copy-Item -Path $SystemFfmpeg -Destination $FfmpegDest -Force
        Write-Host "[+] Copied system FFmpeg ($SystemFfmpeg) -> $FfmpegDest" -ForegroundColor Green
    } else {
        Write-Error "FFmpeg binary not found on system. Please place ffmpeg.exe into bin/ffmpeg.exe."
    }
} else {
    Write-Host "[+] FFmpeg binary present: $FfmpegDest" -ForegroundColor Green
}

# 3. Check bundled font: consola.ttf
$FontsDir = Join-Path $ScriptDir "fonts"
$FontDest = Join-Path $FontsDir "consola.ttf"
if (-not (Test-Path $FontDest)) {
    Write-Host "[*] fonts/consola.ttf not found. Attempting to locate on system..." -ForegroundColor Yellow
    $SysFont = "C:\Windows\Fonts\consola.ttf"
    if (Test-Path $SysFont) {
        New-Item -ItemType Directory -Force -Path $FontsDir | Out-Null
        Copy-Item -Path $SysFont -Destination $FontDest -Force
        Write-Host "[+] Copied system font ($SysFont) -> $FontDest" -ForegroundColor Green
    } else {
        Write-Error "Consolas font not found at $SysFont. Please place consola.ttf into fonts/consola.ttf."
    }
} else {
    Write-Host "[+] Monospace font present: $FontDest" -ForegroundColor Green
}

# 4. Clean previous builds if requested
if ($Clean) {
    Write-Host "[*] Cleaning build/ and dist/ directories..." -ForegroundColor Yellow
    if (Test-Path "$ScriptDir\build") { Remove-Item -Recurse -Force "$ScriptDir\build" }
    if (Test-Path "$ScriptDir\dist") { Remove-Item -Recurse -Force "$ScriptDir\dist" }
}

# 5. Build PyInstaller arguments
$ModeFlag = if ($Mode -eq "onefile") { "--onefile" } else { "--onedir" }

$PyInstallerArgs = @(
    "-m", "PyInstaller",
    $ModeFlag,
    "--name", "binvid",
    "--noconfirm",
    "--add-data", "bin\ffmpeg.exe;bin",
    "--add-data", "fonts\consola.ttf;fonts",
    "--collect-all", "gradio",
    "--collect-all", "gradio_client",
    "--collect-all", "safehttpx",
    "--collect-all", "groovy",
    "--copy-metadata", "tqdm",
    "--copy-metadata", "filelock",
    "--copy-metadata", "packaging",
    "--copy-metadata", "huggingface_hub",
    "binvid\app.py"
)

Write-Host "[*] Executing PyInstaller ($Mode mode)..." -ForegroundColor Yellow
& $PythonExe @PyInstallerArgs

if ($LASTEXITCODE -ne 0) {
    Write-Error "PyInstaller build failed with exit code $LASTEXITCODE."
}

Write-Host "================================================" -ForegroundColor Green
Write-Host "   Build completed successfully!" -ForegroundColor Green
if ($Mode -eq "onefile") {
    Write-Host "   Executable: dist\binvid.exe" -ForegroundColor Green
} else {
    Write-Host "   Distribution folder: dist\binvid\" -ForegroundColor Green
    Write-Host "   Executable: dist\binvid\binvid.exe" -ForegroundColor Green
}
Write-Host "================================================" -ForegroundColor Green
