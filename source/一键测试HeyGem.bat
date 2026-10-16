@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set SSLKEYLOGFILE=
cd /d "%~dp0"
python tools\test_heygem.py
if errorlevel 9009 echo Python not found: install Python 3.10+ and tick "Add to PATH".
pause
