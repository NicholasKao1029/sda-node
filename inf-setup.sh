python3.10 -m venv venv && 
source venv/bin/activate && 
pip install -r ../TensorRT/demo/Diffusion/requirements.txt && # TODO: move requirements from this repo to here
pip install -r requirements.txt && 
pip install --upgrade pip && 
pip install --upgrade tensorrt && 
pip install gunicorn