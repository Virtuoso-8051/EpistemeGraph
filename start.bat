@echo off
REM Double-click to start EpistemeGraph. The browser opens by itself as soon as the server is ready.
REM If it is already running, this just opens it again.
cd /d "%~dp0"
venv\Scripts\python.exe app.py
pause
