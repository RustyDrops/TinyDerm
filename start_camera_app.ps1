$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

Write-Host "Starting TinyDerm Real-Time Camera Scanner on http://127.0.0.1:8000 ..." -ForegroundColor Cyan
Start-Process "http://127.0.0.1:8000"

$PythonExe = Join-Path $ScriptDir ".venv\Scripts\python.exe"
if (Test-Path $PythonExe) {
    & $PythonExe camera_app\server.py
} else {
    python camera_app\server.py
}
