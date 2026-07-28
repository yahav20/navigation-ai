# One-shot setup + run for Windows.
# Installs Python and Node dependencies (first run only), then starts the
# LangGraph backend (http://127.0.0.1:2024) and the atlas-web UI
# (http://localhost:3000). Ctrl+C stops both.
#
# Run from the project root:
#   powershell -ExecutionPolicy Bypass -File .\quick_install.ps1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# -- Prerequisites ------------------------------------------------------------
if (-not (Test-Path ".env")) {
    Write-Host "ERROR: no .env file found in the project root." -ForegroundColor Red
    Write-Host "Copy .env.example to .env and fill in the API keys, then re-run."
    exit 1
}

$pyExe = $null
$pyArgs = @()
if (Get-Command py -ErrorAction SilentlyContinue) {
    & py -3.12 -c "pass" 2>$null
    if ($LASTEXITCODE -eq 0) { $pyExe = "py"; $pyArgs = @("-3.12") }
    else { $pyExe = "py"; $pyArgs = @("-3") }
}
elseif (Get-Command python -ErrorAction SilentlyContinue) {
    $pyExe = "python"
}
if (-not $pyExe) {
    Write-Host "ERROR: Python not found. Install Python 3.12 from https://www.python.org/downloads/" -ForegroundColor Red
    exit 1
}
if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: npm not found. Install Node.js 18+ from https://nodejs.org/" -ForegroundColor Red
    exit 1
}

# -- Install (skipped when already done) --------------------------------------
if (-not (Test-Path ".venv\Scripts\langgraph.exe")) {
    Write-Host ">> Creating virtualenv and installing Python dependencies (takes a few minutes)..."
    & $pyExe @pyArgs -m venv .venv
    & ".venv\Scripts\pip.exe" install -r requirements-server.txt
    if ($LASTEXITCODE -ne 0) { Write-Host "ERROR: pip install failed." -ForegroundColor Red; exit 1 }
}

if (-not (Test-Path "atlas-web\node_modules")) {
    Write-Host ">> Installing web UI dependencies..."
    Push-Location atlas-web
    npm install --no-audit --no-fund
    if ($LASTEXITCODE -ne 0) { Pop-Location; Write-Host "ERROR: npm install failed." -ForegroundColor Red; exit 1 }
    Pop-Location
}

# -- Run -----------------------------------------------------------------------
Write-Host ">> Starting backend (LangGraph dev server) on http://127.0.0.1:2024 ..."
$backend = Start-Process -FilePath "$PSScriptRoot\.venv\Scripts\langgraph.exe" `
    -ArgumentList "dev", "--allow-blocking", "--no-browser" `
    -WorkingDirectory $PSScriptRoot -PassThru -NoNewWindow

Write-Host ">> Waiting for the backend to come up..."
$backendUp = $false
for ($i = 0; $i -lt 60; $i++) {
    try {
        Invoke-WebRequest -UseBasicParsing -Uri "http://127.0.0.1:2024/ok" -TimeoutSec 2 | Out-Null
        $backendUp = $true
        break
    }
    catch {
        if ($backend.HasExited) {
            Write-Host "ERROR: backend exited during startup - see the log output above." -ForegroundColor Red
            exit 1
        }
        Start-Sleep -Seconds 2
    }
}
if (-not $backendUp) {
    Write-Host "ERROR: backend did not respond on http://127.0.0.1:2024 within 2 minutes." -ForegroundColor Red
    Stop-Process -Id $backend.Id -Force -ErrorAction SilentlyContinue
    exit 1
}

Write-Host ""
Write-Host ">> Backend is up. Starting the web UI ..."
Write-Host ">> Open http://localhost:3000 in your browser once Next.js reports 'Ready'."
Write-Host ">> Press Ctrl+C to stop both servers."
Write-Host ""
try {
    Set-Location atlas-web
    npm run dev
}
finally {
    Set-Location $PSScriptRoot
    if ($backend -and -not $backend.HasExited) {
        Stop-Process -Id $backend.Id -Force -ErrorAction SilentlyContinue
    }
}
