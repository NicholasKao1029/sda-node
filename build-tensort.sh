# install python3.10 and create a venv
sudo apt update && sudo apt upgrade -y
sudo apt install software-properties-common -y
sudo add-apt-repository --yes ppa:deadsnakes/ppa

sudo apt update && sudo apt install python3.10 python3.10-venv python3.10-dev -y

# create and enable the venv
python3.10 -m venv env
source env/bin/activate

# Install system TensorRT
sudo apt install tensorrt tensorrt-dev tensorrt-devel tensorrt-libs -y

# Clone the TensorRT repo
git clone https://github.com/NVIDIA/TensorRT
cd TensorRT
git submodule update --init --recursive

pip install --upgrade pip
pip install --upgrade tensorrt

export TRT_OSSPATH=$PWD

cd $TRT_OSSPATH
mkdir -p build && cd build
cmake .. -DTRT_OUT_DIR=$PWD/out
cd plugin
make -j$(nproc)

export PLUGIN_LIBS="$TRT_OSSPATH/build/out/libnvinfer_plugin.so"

cd $TRT_OSSPATH/demo/Diffusion
pip install -r requirements.txt