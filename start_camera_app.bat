@echo off
cd /d "%~dp0"
echo Starting TinyDerm Real-Time Camera Scanner on http://127.0.0.1:8000 ...
start http://127.0.0.1:8000
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" camera_app\server.py
) else (
    python camera_app\server.py
)
pause
