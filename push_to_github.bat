@echo off
echo ===================================================
echo 🚀 Push Docker-Cloud Updates to GitHub
echo ===================================================
echo.
set /p TOKEN="Enter your GitHub Personal Access Token (or password): "
if "%TOKEN%"=="" (
    echo ❌ Token cannot be empty.
    pause
    exit /b
)

echo.
echo 📤 Pushing to https://github.com/gp3ci/Docker-cloud.git ...
"c:\Users\Netcom\Desktop\Docker-cloud-main\mingit\cmd\git.exe" push -u "https://%TOKEN%@github.com/gp3ci/Docker-cloud.git" main

echo.
echo ===================================================
echo ✅ Done!
echo ===================================================
pause
