#!/bin/bash

# assumes python3.10 is installed
# assumes env var TRT_OSSPATH is set to the path of the TensorRT OSS repo

python3.10 -m venv venv && 
source venv/bin/activate && 
pip install -r tensorrt-requirements &&
pip install -r requirements.txt && 
pip install --upgrade pip && 
pip install --upgrade tensorrt && 
pip install gunicorn