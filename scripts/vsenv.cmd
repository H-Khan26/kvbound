@echo off
rem Runs a command inside the MSVC x64 build environment: scripts\vsenv.cmd cmake --preset cuda
setlocal EnableDelayedExpansion
where cl >nul 2>nul && goto run
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
for /f "usebackq delims=" %%i in (`""!VSWHERE!" -latest -products * -property installationPath"`) do set "VSROOT=%%i"
if not defined VSROOT echo vsenv: no Visual Studio installation found 1>&2 & exit /b 1
call "!VSROOT!\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
:run
%*
