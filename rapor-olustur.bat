@echo off
rem AutoFcc Gorev Raporu - VERIYI ICINE GOMULU HTML uretir (ag gerektirmez)
cd /d "%~dp0"
.venv\Scripts\python.exe tools\dashboard.py --static gorev-raporu.html
echo.
echo Rapor hazir: gorev-raporu.html (cift tikla, tarayicida acilir)
pause
