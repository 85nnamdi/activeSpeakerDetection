#!/bin/bash
SET WGET_EXE=..\3rdparty\windows\wget\wget.exe
SET OPENPOSE_URL=https://github.com/CMU-Perceptual-Computing-Lab/openpose/releases/download/v1.6.0/openpose-1.6.0-binaries-win64-gpu-flir-3d_recommended.zip
SET OPENPOSE_FOLDER=openpose/

echo:
echo ------------------------- Downloading Openpose 1.6 -------------------------
set BODY_25_FOLDER=%OPENPOSE_FOLDER%body_25/
%WGET_EXE% -c %OPENPOSE_URL% -P %OPENPOSE_FOLDER%

unzip -d ./openpose/ openpose-1.6.0-binaries-win64-gpu-flir-3d_recommended.zip
