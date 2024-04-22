@echo off
:start
echo Checking if server is running...
tasklist /FI "IMAGENAME eq python.exe" 2>NUL | find /I /N "python.exe">NUL
if "%ERRORLEVEL%"=="0" (
    echo Server is running.
) else (
    echo Server is not running. Restarting...
    start "" python manage.py runserver
)
timeout /t 10 /nobreak >NUL
goto start
