$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

$PythonExe = Join-Path $ScriptDir ".venv\Scripts\python.exe"
if (Test-Path $PythonExe) {
    & $PythonExe run.py
} else {
    python run.py
}
