python3.10 -m venv compiler-venv && 
source compiler-venv/bin/activate &&
pip install -r requirements.txt &&
pip install -r tensorrt-requirements.txt &&
pip install --upgrade pip && 
pip install --upgrade tensorrt && 
pip install diffusers==0.7.2
pip install nvidia-pyindex
pip install polygraphy
pip install transformers
pip install scipy