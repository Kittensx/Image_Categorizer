@echo off
cd /d "%~dp0" 
call venv\Scripts\activate
call python.exe runcat.py
pause
