@echo off
title MikoTim Sentinel Pro
color 0A
echo ========================================================
echo    MikoTim Sentinel Pro - AI Security Monitor
echo ========================================================
echo.
echo Starting Sentinel server...
echo Web console will open at http://127.0.0.1:5000
echo.

start "" "http://127.0.0.1:5000"
"C:\Users\Lenovo\AppData\Local\Python\pythoncore-3.14-64\python.exe" app.py

pause
