git clone https://github.com/NVIDIA/TensorRT && 
cd TensorRT &&
git submodule update --init --recursive
apt update && apt upgrade -y &&
apt install software-properties-common -y &&
add-apt-repository --yes ppa:deadsnakes/ppa &&

# 3.8 is the default python version set in the above image
apt update && apt install python3.8 python3.8-venv python3.8-dev -y &&
#3.10 is what is used by sda-node
apt update && apt install python3.10 python3.10-venv python3.10-dev -y &&
# create and enable the venv
python3.8 -m venv env &&
source env/bin/activate &&
# OR do 3.10
python3.10 -m venv env && source env/bin/activate &&
apt install tensorrt tensorrt-dev tensorrt-libs -y &&
python3.10 -m pip install --upgrade pip &&
python3.10 -m pip install --upgrade tensorrt &&
# compile tensorrt plugin
export TRT_OSSPATH=$PWD &&

# compile, this caused lots of issues when other images were not specified
# due to lots of ENVS not being sent I believe
cd $TRT_OSSPATH &&
mkdir -p build && cd build &&
cmake .. -DTRT_OUT_DIR=$PWD/out &&
cd plugin &&
make -j$(nproc) &&
# path to compiled plugin, required for tensorRT inference
export PLUGIN_LIBS="$TRT_OSSPATH/build/out/libnvinfer_plugin.so" &&



# this is for demo AND it turns out required for SDA-NODE
cd $TRT_OSSPATH/demo/Diffusion &&
pip install -r requirements.txt &&

# this is where the MODEL compiles down
mkdir -p onnx engine output &&

# just need read permission
export HF_TOKEN=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 &&
LD_PRELOAD=${PLUGIN_LIBS} python3 demo-diffusion.py "a beautiful photograph of Mt. Fuji during cherry blossom" --hf-token=$HF_TOKEN -v 


git clone https://github.com/NicholasKao1029/sda-node.git
cd sda-node
