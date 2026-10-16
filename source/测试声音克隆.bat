@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set SSLKEYLOGFILE=
cd /d "%~dp0"
python tools\test_voice.py
pause
