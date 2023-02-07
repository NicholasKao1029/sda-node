python3.10 -m venv compiler-venv && 
source compiler-venv/bin/activate &&
pip install -r sda-node/requirements.txt &&
pip install -r TensorRT/demo/Diffusion/requirements.txt &&
pip install --upgrade pip && 
pip install --upgrade tensorrt && 
pip install diffusers==0.7.2