@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set SSLKEYLOGFILE=
cd /d "%~dp0"
python -m app.launch
if errorlevel 9009 echo Python not found: install Python 3.10+ and tick "Add to PATH".
pause
