#!/bin/bash
.\\build\\bin\\OpenPoseDemo.exe --video examples\\media\\youtube.mp4 --face --hand --write_json dataset/output_json_folder/

read name
echo "Hello $name"