@echo off
REM Launch the GramSynth backend with the MSVC + CUDA build environment so the
REM StyleGAN custom CUDA kernels compile/load (fast path) for jobs it spawns.
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1"
set "CUDA_PATH=%CUDA_HOME%"
set "PATH=%CUDA_HOME%\bin;%PATH%"
cd /d "%~dp0"
".\.venv\Scripts\python.exe" -m uvicorn main:app --port 8000
