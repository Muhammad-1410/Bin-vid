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

$PythonExe = Join-Path $ScriptDir "venv\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
    $PythonExe = (Get-Command python -ErrorAction SilentlyContinue).Source
    if (-not $PythonExe) {
        Write-Error "Python executable not found. Please activate your virtual environment or install Python."
    }
}
Write-Host "[+] Using Python: $PythonExe" -ForegroundColor Green

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

if ($Clean) {
    Write-Host "[*] Cleaning build/ and dist/ directories..." -ForegroundColor Yellow
    if (Test-Path "$ScriptDir\build") { Remove-Item -Recurse -Force "$ScriptDir\build" }
    if (Test-Path "$ScriptDir\dist") { Remove-Item -Recurse -Force "$ScriptDir\dist" }
}

$ModeFlag = if ($Mode -eq "onefile") { "--onefile" } else { "--onedir" }

$PyInstallerArgs = @(
    "-m", "PyInstaller",
    $ModeFlag,
    "--name", "binvid",
    "--noconfirm",
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
