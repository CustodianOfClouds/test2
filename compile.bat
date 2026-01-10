@echo off
setlocal

REM ================================
REM Configuration
REM ================================
set COMPILE_DIR=compile
set BUILD_DIR=build
set APP_NAME=Mars
set VERSION=4.5.54111
set JAR_NAME=%APP_NAME%-%VERSION%.jar
set VENDOR=ccp

REM ================================
REM Step 0: Clean previous outputs
REM ================================
if exist "%COMPILE_DIR%" rmdir /s /q "%COMPILE_DIR%"

REM ================================
REM Step 1: Compile Java sources
REM ================================
echo === Step 1: Compile Java sources ===
mkdir "%COMPILE_DIR%"

javac --release 10 ^
  -d "%COMPILE_DIR%" ^
  @java_files.txt
if errorlevel 1 goto :error

REM ================================
REM Step 2: Prepare build directory
REM ================================
echo.
echo === Step 2: Prepare build directory ===
if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"

REM ================================
REM Step 3: Create versioned JAR in /build
REM ================================
echo.
echo === Step 3: Create JAR ===
jar cmf mainclass.txt "%BUILD_DIR%\%JAR_NAME%" ^
  -C "%COMPILE_DIR%" . ^
  PseudoOps.txt ^
  Config.properties ^
  Syscall.properties ^
  Settings.properties ^
  MipsXRayOpcode.xml ^
  registerDatapath.xml ^
  controlDatapath.xml ^
  ALUcontrolDatapath.xml ^
  help ^
  images
if errorlevel 1 goto :error

REM ================================
REM Step 4: Build Windows EXE into /build
REM ================================
echo.
echo === Step 4: Build Windows EXE ===
jpackage ^
  --input "%BUILD_DIR%" ^
  --dest "%BUILD_DIR%" ^
  --name "%APP_NAME%" ^
  --icon "%CD%\images\mars.ico" ^
  --main-jar "%JAR_NAME%" ^
  --type exe ^
  --app-version "%VERSION%" ^
  --vendor "%VENDOR%" ^
  --win-menu ^
  --verbose
if errorlevel 1 goto :error

REM ================================
REM Step 5: Cleanup temporary files
REM ================================
echo.
echo === Step 5: Cleanup ===
rmdir /s /q "%COMPILE_DIR%"

echo.
echo === BUILD SUCCESSFUL ===
pause
exit /b 0

:error
echo.
echo === BUILD FAILED ===
pause
exit /b 1