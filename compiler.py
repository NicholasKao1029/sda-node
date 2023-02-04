#This module is meant for direct use only. For API-usage please check SDA-TRAINER.
#Based off NVIDIA's demo
import argparse
from threads.trt.models import CLIP, UNet, VAE
import os
import onnx
import torch
from diffusers import UNet2DConditionModel, AutoencoderKL, __version__
from transformers import CLIPTextModel
from threads.trt.utilities import Engine
import argparse
import io
from termcolor import colored
from huggingface_hub import create_repo, HfApi, utils, hf_hub_download, login
import json
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', required=True, help="Local Path to folder or HuggingFace ID to the diffuser model")
parser.add_argument('-o', '--output', default="./output", help="Output directory")
parser.add_argument('--build-dynamic-shape', action='store_true', help="Build TensorRT engines with dynamic image shapes.")
parser.add_argument('--hf-token', type=str, default="none", help="HuggingFace API access token for downloading model checkpoints")
parser.add_argument('-v', '--verbose', action='store_true', help="Enable Verbose")
parser.add_argument('-d', '--disable-folder-check', action='store_true', help="Disable existing folder check")
parser.add_argument('-s', '--skip-compiling', action='store_true', help="Skip compiling")
args = parser.parse_args()

if args.skip_compiling is not True:
    if str(__version__) != "0.7.2":
        raise ImportError(f"""
    Diffusers version is not 0.7.2, instead it's {__version__}
    This will raise errors during conversion of the UNET
    Please download version 0.7.2 by running the following command:
    python3 -m pip install diffusers==0.7.2
        """
        )

def getModelPath(name, onnx_dir, opt=True):
    return os.path.join(onnx_dir, name+('.opt' if opt else '')+'.onnx')

def select_option(start, options):
    print(start)
    while True:
        for i, option in enumerate(options):
            print(f"{i+1}. {option}")
        choice = input("Select an option:")
        try:
            choice = int(choice)
            if 1 <= choice <= len(options):
                print(colored(f"You selected {options[choice-1]}", 'green'))
                return options[choice-1]
            else:
                print(colored("Invalid selection. Please choose a number between 1 and", 'red'), len(options))
        except ValueError:
            print(colored("Invalid selection. Please enter a number.", 'red'))

trt_version = "none"
cuda_version = "none"
cudnn_version = "none"
onnx2trt_version = "none"

plugin_path = os.environ['PLUGIN_LIBS']
build_path = os.path.abspath(os.path.join(os.path.dirname(plugin_path), os.pardir))
cmakecache_path = os.path.join(build_path, 'CMakeCache.txt')

if os.path.exists(cmakecache_path):
    with open(cmakecache_path) as f:
        print("THE FOLLOWING VERSIONS WERE EXTRACTED FROM THE CMAKECACHE USED TO BUILD THE GIVEN PLUGIN.")
        for line in f:
            if "CMAKE_PROJECT_VERSION:STATIC" in line:
                trt_version = line.split("=")[-1].replace("\n","")
                print(f"Detected TensorRT version: {trt_version}")
            if "CUDA_VERSION:UNINITIALIZED" in line:
                cuda_version = line.split("=")[-1].replace("\n","")
                print(f"Detected CUDA version: {cuda_version}")
            if "CUDNN_VERSION:UNINITIALIZED" in line:
                cudnn_version = line.split("=")[-1].replace("\n","") #<-- aka compute version
                print(f"Detected CUDNN version: {cudnn_version}")
            if "ONNX2TRT_VERSION:STRING" in line:
                onnx2trt_version = line.split("=")[-1].replace("\n","")
                print(f"Detected ONNX2TRT version: {onnx2trt_version}")
else:
    print("Failed to detect CMakeCache.txt file. If you know what compute version your plugin.so is using, please type them")
    print("This is to allow other users to use the model with proper compute versioning.")
    print("This is crucial if you want to upload to huggingface.")
    opt_notfound = select_option("Do you know the compute version?", ['Yes', 'No'])
    if opt_notfound.lower() == 'yes':
        print("Type the Compute/CUDNN version in the following format: X.x example: 7.5")
        cudnn_version = input("Type: ")
        print("Compute/CUDNN version set. TensorRT, CUDA, and ONNX2TRT versions have not been configured, but they are not necessary.")
    else:
        print("Generated config file will not display compute version.")

if args.disable_folder_check is not True:
    if os.path.exists(args.output):
        folder_exists = select_option(f"The output folder ({args.output}) already exists. It is possible that there already is a model inside of it. Abort? or delete and continue?",
        ['Abort', 'Delete & Continue'])
        print(folder_exists)
        if folder_exists == "Abort":
            exit()
        elif folder_exists == "Delete & Continue":
            shutil.rmtree(args.output)

onnx_dir = os.path.join(args.output, 'onnx')
engine_dir = os.path.join(args.output, 'engine')
os.makedirs(onnx_dir, exist_ok=True)
os.makedirs(engine_dir, exist_ok=True)

hf_token = args.hf_token
device = "cuda"
verbose = args.verbose

max_batch_size = 16 if args.build_dynamic_shape is False else 4
opt_batch_size = max_batch_size
denoising_fp16 = True

if args.skip_compiling is not True:
    models = {
        'clip': CLIP(hf_token=hf_token, device=device, verbose=verbose, max_batch_size=max_batch_size),
        'unet': UNet(hf_token=hf_token, fp16=denoising_fp16, device=device, verbose=verbose, max_batch_size=max_batch_size),
        'de_vae': VAE(hf_token=hf_token, device=device, verbose=verbose, max_batch_size=max_batch_size),
        # # 'en_vae': EN_VAE(hf_token=hf_token, device=device, verbose=verbose, max_batch_size=max_batch_size),
    }
else:
    models = {}

#Note:
# en_vae is for encoding (for img2img)
# de_vae is for decoding (for img2img and txt2img)

def get_model(type, path):
    if type in 'unet':
        #UNET
        tmp_model = UNet2DConditionModel.from_pretrained(
            path, 
            subfolder="unet",
            use_auth_token=hf_token,
            torch_dtype = torch.float16
        ).to(device)
    elif type == 'clip':
        #CLIP
        tmp_model = CLIPTextModel.from_pretrained(
            path,
            subfolder="text_encoder",
            use_auth_token=hf_token,
        ).to(device)
    elif type == 'de_vae':
        #DECODE VAE
        tmp_model = AutoencoderKL.from_pretrained(
            path,
            subfolder="vae",
            use_auth_token=hf_token,
        ).to(device)
        tmp_model.forward = tmp_model.decode
    return tmp_model

#Just to fill
opt_image_height = 512
opt_image_width = 512

#check this later
onnx_opset = 16

print("Using model:", args.model)

for model_name, obj in models.items():
    engine = Engine(model_name, engine_dir)
    onnx_path = getModelPath(model_name, onnx_dir, opt=False)
    onnx_opt_path = getModelPath(model_name, onnx_dir, opt=True)

    print(f"Exporting model: {onnx_path}")
    #important: change model path to desired one
    model = get_model(model_name, args.model)
    #opt_batch_size does not necesairly means that it's going to be static batch size
    with torch.inference_mode(), torch.autocast("cuda"):
        inputs = obj.get_sample_input(opt_batch_size, opt_image_height, opt_image_width)
        torch.onnx.export(model,
                inputs,
                onnx_path,
                export_params=True,
                opset_version=onnx_opset,
                do_constant_folding=True,
                input_names = obj.get_input_names(),
                output_names = obj.get_output_names(),
                dynamic_axes=obj.get_dynamic_axes(),
        )
    print(f"Generating optimized ONNX model: {onnx_opt_path}")
    #minimal optimization flag was removed here for obvious reasons
    onnx_opt_graph = obj.optimize(onnx.load(onnx_path))
    onnx.save(onnx_opt_graph, onnx_opt_path)
    # Build engine
    print(f"Generating TensorRT model: {onnx_opt_path}")
    # Disable preview since it requires high levels of TRT version
    engine.build(onnx_opt_path, fp16=True , \
        input_profile=obj.get_input_profile(opt_batch_size, opt_image_height, opt_image_width, \
            static_batch=False, static_shape=not args.build_dynamic_shape), \
        enable_preview=False)

shutil.rmtree(onnx_dir)

option = select_option("Upload model to HuggingFace?", ['Y', 'N'])
if option.lower() == "y":

    loggedprev = select_option("Have you logged previously to huggingface?", ['Y', 'N'])
    if loggedprev.lower() == 'n':
        login()

    mkrepo = select_option("Create or use an existing repo?", ['CREATE', 'EXISTING'])
    print("The name MUST include your username. For ex.: chavinlo/AlienPop")
    repo_name = input("Repository Name:")
    #kinda confusing
    if mkrepo == 'CREATE':
        priv_opt = select_option("Make it private?", ['Y', 'N'])
        try:
            create_repo(repo_name, private=True if priv_opt.lower() == 'y' else False, repo_type="model")
        except utils._errors.HfHubHTTPError:
            print("Skipping...")
    
    path_in_repo = "engine/"
    cuspath = select_option("By default the model will be uploaded on /engine, do you want to change this?", ['Y', 'N'])
    if cuspath.lower() == 'y':
        path_in_repo = input("Custom path:")
    
    revision = "main"
    cusrev = select_option("By default the model will be uploaded on the main branch, do you want to change this?", ['Y', 'N'])
    if cusrev.lower() == 'y':
        revision = input("Custom branch/revision:")

    print("The following will be the configuration file. This has been generated and is highly recommended to not edit it.")
    config = {
        "_class_name": "StableDiffusionAccelerated_Base",
        "_sda_version": "0.1.2",
        "_trt_version": trt_version,
        "_cuda_version": cuda_version,
        "_cudnn_version": cudnn_version,
        "_onnx2trt_version": onnx2trt_version,
        "unet": {
            "precision": "fp16",
            "path": f"{path_in_repo}unet.plan"
        },
        "clip": {
            "path": f"{path_in_repo}clip.plan"
        },
        "de_vae": {
            "path": f"{path_in_repo}de_vae.plan"
        }
    }
    print(config)

    #Model Card
    def_readme = """
---
tags:
- TensorRT
- Text2Image
- Stable Diffusion
- Image2Image
- SDA
---

# {MODEL_NAME} converted into TensorRT

<a href="https://github.com/chavinlo/sda-node/"><img src="https://i.imgur.com/fQS926g.png"></a>

Model converted from diffusers into TensorRT for accelerated inference up to 4x faster.

For how to use the model check https://github.com/chavinlo/sda-node

This model was automatically converted by SDA-node

Compilation configuration:

    """

    scheduler_config = hf_hub_download(repo_id=args.model, filename="scheduler_config.json", subfolder="scheduler")
    def_readme = def_readme.format(MODEL_NAME=args.model)
    config_json = json.dumps(config, indent=4)
    def_readme += "\n\n```json\n" + config_json + "\n```"
    os.makedirs(os.path.join(args.output, 'scheduler'), exist_ok=True)
    open(os.path.join(args.output, 'README.md'), "w").write(def_readme)
    open(os.path.join(args.output, 'model_index.json'), "w").write(config_json)
    open(os.path.join(os.path.join(args.output, 'scheduler'), 'scheduler_config.json'), "w").write(open(scheduler_config).read())
    
    print("Uploading...")
    api = HfApi()
    api.upload_folder(
        folder_path=args.output,
        path_in_repo="",
        repo_id=repo_name,
        repo_type="model"
    )
    print("\n\n")
    print("Successfully uploaded")
    print(f"Uploaded into https://huggingface.co/{repo_name}")

print(f"Your model is available at: {os.path.abspath(engine_dir)}")