#!/bin/bash

# assumes python3.10 is installed
# 3.10.8 works
# 3.10.5 works
# assumes env var TRT_OSSPATH is set to the path of the TensorRT OSS repo

python3.10 -m venv venv && 
source venv/bin/activate && 
pip install -r tensorrt-requirements.txt &&
pip install -r requirements.txt && 
pip install --upgrade pip && 
pip install --upgrade tensorrt && 
pip install nvidia-pyindex
pip install polygraphy
pip install transformers
pip install cuda-python
pip install scipy
pip install gunicorn
# why are these doubled up? IDK 
pip install diffusers
pip install tensorrt
