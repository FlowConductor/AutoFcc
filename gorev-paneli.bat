@echo off
rem AutoFcc Gorev Paneli - SADECE LOKAL (127.0.0.1), internete acilmaz
cd /d "%~dp0"
.venv\Scripts\python.exe tools\dashboard.py
pause
