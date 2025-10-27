@echo off
REM create_desktop_shortcut.bat
REM Create a desktop shortcut for the DQN GridWorld app

setlocal

set SCRIPT_DIR=%~dp0
set LAUNCHER=%SCRIPT_DIR%Launch_DQN_GridWorld.bat
set DESKTOP=%USERPROFILE%\Desktop
set SHORTCUT_NAME=DQN GridWorld Visualizer.lnk

REM Create VBScript to generate shortcut
set VBS_FILE=%TEMP%\create_shortcut.vbs
echo Set oWS = WScript.CreateObject("WScript.Shell") > "%VBS_FILE%"
echo sLinkFile = "%DESKTOP%\%SHORTCUT_NAME%" >> "%VBS_FILE%"
echo Set oLink = oWS.CreateShortcut(sLinkFile) >> "%VBS_FILE%"
echo oLink.TargetPath = "%LAUNCHER%" >> "%VBS_FILE%"
echo oLink.WorkingDirectory = "%SCRIPT_DIR%" >> "%VBS_FILE%"
echo oLink.Description = "Launch the DQN GridWorld Streamlit Visualizer" >> "%VBS_FILE%"
echo oLink.Save >> "%VBS_FILE%"

REM Execute VBScript
cscript //nologo "%VBS_FILE%"

REM Cleanup
del "%VBS_FILE%"

echo.
echo Desktop shortcut created at: %DESKTOP%\%SHORTCUT_NAME%
echo.
pause
