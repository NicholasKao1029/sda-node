# This project has been archived.

-------

<img src="https://i.imgur.com/fQS926g.png"></a>
## Stable Diffusion Accelerated
## 60 steps per second!
### [Special thanks to Redmond AI for providing compute for testing](http://www.redmond.ai/)

A demo is available on our discord server! https://discord.gg/8Sh2T6gjd2 on #text2img channel.

This is the Demo Base Node module for Stable Diffusion Accelerated. Using TensorRT, we can achieve speeds up to 4 times faster than native PyTorch. 

# Notice

Note that, this only includes the basics and has been made to serve as an example on how extensible TensorRT can be while combining it with SD.

If you want a more extended, advanced, version, with support for multi-model serving, scalable, and modular models, please send us a message to dep#2171 on discord.

# Description

Based on the Demo provided by NVIDIA, we (I) extended it's capabilities, some of them are:

* API for inference
* Weighted prompts
* More Schedulers
* Benchmarking
* More Step Counts

## How it works & where to get models

TensorRT optimizes the SD model by compiling it into a highly optimized version that can be run on NVIDIA GPUs. This adds some limitations such as limited batch size and resolutions (up to 1024px). It optimizes the CLIP, UNET, and VAE.

You can download pre-compiled models from our HuggingFace' TensorRT repository:
https://huggingface.co/tensorrt

# Usage:

## Instalation:
In the meantime, this software is API only. If you have JS and HTML skills, a demo page would really be appreciated!

To initiate the API server, you need to first install TensorRT and it's dependencies. I have made a small shell script to install most of the requirements, but it's not bulletproof:

```
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
```

Now you should have the dependencies and plugin compiled. I highly recommend running `echo $PLUGIN_LIBS` and save the output somewhere, as this is the compiled plugin needed for tensorRT inference

now go back to the SDA-Node repo or clone it if you haven't.

```
git clone https://github.com/chavinlo/sda-node
cd sda-node
```

now just install your python server of choice and start the server:
```
pip install -r requirements.txt
pip install gunicorn
LD_PRELOAD=${PLUGIN_LIBS} gunicorn -w 1 -b 0.0.0.0:5000 main:app
```

It is EXTREMELY important that you run it with `LD_PRELOAD=${PLUGIN_LIBS}` to use the needed plugins.

## Inference

By default, it will use the configuration file on cfg/basic.json

just change "model_path" to the folder where your .plan files are available.

For example, if I want to use Anything-V3: https://huggingface.co/tensorrt/Anything-V3

First, clone the HuggingFace Repo:
```
# Install git-lfs first
sudo apt install git-lfs
git lfs install

git clone https://huggingface.co/tensorrt/Anything-V3
```

You can also download each file manually on the engine/ folder, just place them back in one folder again.

Then, go to basic.json and edit the following line:
`"model_path": "/workspace/TensorRT/demo/Diffusion/Anything-V3/engine",`
Into:
`"model_path": "anything-v3/engine",`
It must be changed to the path of where the *.plan files are at.

After this, just start the server again, and the API will be available at 127.0.0.1:5000, which you can use like below:

## Text2Image

Send a JSON request in the following format:
```
{
	"prompt": str,
	"negprompt": str,
	"width": int (min. 256, max. 1024),
	"height": int (min. 256, max. 1024),
	"steps": int (multiple of 5, min. 25, check config),
	"cfg": float,
	"seed": int (-1 for random),
	"scheduler": str,
	"mode": str,
	"lpw": bool
}
```

Where:

* prompt: is the prompt, but if you want to use weighting "()" or "[]" you have to enable LPW
* negative prompt: same as the prompt, but for things you want to avoid
* width: integer, minimun 256, maximun 1024. Large changes might generate a mismatch error, just try again
* height: same as width
* steps: Depends on the config, if using the default, make sure to use a multiple of 5, minimun 25
* cfg: float, how much text influences
* seed: integer, -1 for random seed
* scheduler: str, choose between EULER-A, LMSD, DPMS. DPMS and LMSD are accelerated, where as EULER-A is imported from diffusers (slower)
* lpw: bool, wether to enable Long and Extended Prompt module (where it will accept weighting) or to disable it (faster)
* mode: str, "file" to return a raw file download (instantly viewable), or "json" to return a json like the following:

Upon success:
```
{
    "status": "done",
    "content": {
        "img": image base64 encoded UTF-8,
        "time": time taken to process in seconds
    }
}
```

Upon failure:
```
{
    "status": "fail",
    "content": str with reason for failure
}
```

# Compilation 

This example
- set up the compiler env via `compiler-setup.sh` 
- `$HF_TOKEN` is set get from [huggingface](https://huggingface.co/settings/tokens)
- `$PLUGIN_LIBS` is set

```
 LD_PRELOAD=${PLUGIN_LIBS} python compiler.py -m nuigurumi/basil_mix --build-dynamic-shape --hf-token $HF_TOKEN
```

Compilation takes ~20 min A100 80gb


# Benchmark
An extensive list of benchmarks is available at [docs/benchmarks.md](docs/benchmarks.md)

# Examples:
Generated with Anything-V3

## 512px 25 Steps - 0.47s:

<img src="https://i.imgur.com/iaAB3Nq.png" height="512">

```

Sent Request:
{
	"prompt": "(Masterpiece:1.2), best quality, illustration, (delicate details:1.5),extremely detailed CG, lovely layered white hair, absurdly long hair, (glowing blue eyes), lip gloss, makeup, (school:1.5), evil smile, medium breasts, school girl, ((arms behind back)), school uniform",
	"negprompt": "nsfw, (worst quality, low quality:1.3), (depth of field, blurry:1.2), (greyscale, monochrome:1.1), 3D face, nose, cropped, lowres, text, jpeg artifacts, signature, watermark, username, blurry, artist name, trademark, watermark, title",
	"width": 512,
	"height": 512,
	"steps": 25,
	"cfg": 7,
	"seed": 86,
	"scheduler": "LMSD",
	"mode": "file",
	"lpw": true
}

Bench:
|      PREP      |      2.00 ms |
|     CLIP**     |     35.00 ms |
|   UNET x 25    |    390.00 ms |
|      VAE*      |      3.00 ms |
|    SERVING     |     42.00 ms |
|    TOTALCOM    |    471.00 ms |
|     TOTAL      |    474.00 ms |
w512 x h512
lpw: True
scheduler: LMSD
accelerated: True
```

## 512px 50 Steps - 0.84s:

<img src="https://i.imgur.com/c05Cl7i.png" height="512">

```

Sent Request:
{
	"prompt": "(extremely detailed CG unity 8k wallpaper,masterpiece, best quality, ultra-detailed,best shadow),(multicolored background),(pop art:1.4),((illustration)),(beautiful detailed face),(floating hair),dynamic angle,High contrast,(limited palette:1.2),(best illumination, an extremely delicate and beautiful)",
	"negprompt": "nsfw, (worst quality, low quality:1.3), (depth of field, blurry:1.2), (greyscale, monochrome:1.1), 3D face, nose, cropped, lowres, text, jpeg artifacts, signature, watermark, username, blurry, artist name, trademark, watermark, title",
	"width": 512,
	"height": 512,
	"steps": 50,
	"cfg": 7,
	"seed": 432,
	"scheduler": "LMSD",
	"mode": "file",
	"lpw": true
}

Bench:
|      PREP      |      2.00 ms |
|     CLIP**     |     25.00 ms |
|   UNET x 50    |    764.00 ms |
|      VAE*      |      3.00 ms |
|    SERVING     |     42.00 ms |
|    TOTALCOM    |    835.00 ms |
|     TOTAL      |    838.00 ms |
w512 x h512
lpw: True
scheduler: LMSD
accelerated: True
```

## 768px 50 Steps - 1.96s:

<img src="https://i.imgur.com/MNnjUxS.png" height="768">

```

Sent Request:
{
	"prompt": "(Masterpiece:1.2), (best quality:1.2), (illustration:1.1), (1girl:1.1), detailed, Cinematic light, intricate detail, highres, a character design of young black lolita dressed girl, grey and blue theme, wavy white long hair by krenz cushart and mucha and akihito yoshida",
	"negprompt": "nsfw, (worst quality, low quality:1.3), (depth of field, blurry:1.2), (greyscale, monochrome:1.1), 3D face, nose, cropped, lowres, text, jpeg artifacts, signature, watermark, username, blurry, artist name, trademark, watermark, title",
	"width": 768,
	"height": 768,
	"steps": 50,
	"cfg": 7,
	"seed": 7011,
	"scheduler": "LMSD",
	"mode": "file",
	"lpw": true
}

Bench:
|      PREP      |      2.00 ms |
|     CLIP**     |     37.00 ms |
|   UNET x 50    |   1803.00 ms |
|      VAE*      |      6.00 ms |
|    SERVING     |    112.00 ms |
|    TOTALCOM    |   1960.00 ms |
|     TOTAL      |   1962.00 ms |
w768 x h768
lpw: True
scheduler: LMSD
accelerated: True
```

# License:
The Software is intended for individual use only. Any use by groups such as companies, large communities, commercial, for-profit entities, etc. must be approved before-hand by the Copyright Owner. This includes but is not limited to: Discord Bots, Software-as-a-Service, etc.
