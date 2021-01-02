from pathlib import Path
import os
import datetime
import yaml
from addict import Dict
import argparse
from utils.utils import env_to_path, tensor_to_PIL
from data.data import *
from dae.dae import DAE
import torch
import numpy as np
import PIL
import torchvision
import matplotlib.pyplot as plt

# -----------------------------
# -----  Parse arguments  -----
# -----------------------------

parser = argparse.ArgumentParser(description="traversals generator")
parser.add_argument(
    "--config",
    type=str,
    default="./config/defaults.yaml",
    help="path to config file",
)
parser.add_argument(
    "--dae-output", action="store_true", help="generate images to see the original image and its reconstruction by the DAE"
)
parser.add_argument(
    "--dae-checkpoint", type=str, default=None, help="dae model checkpoint to use"
)
parser.add_argument(
    "--print-values", action="store_true", help="print output image values to console"
)
parser.add_argument(
    "--single-image", action="store_true", help="use latent space exploration output"
)
args = parser.parse_args()

# -----------------------
# -----  Load opts  -----
# -----------------------

with open(args.config, "r") as f:
    opts = yaml.safe_load(f)

opts = Dict(opts)
opts.output_path = str(env_to_path(opts.output_path))
print("Config output_path:", opts.output_path)

loader = get_loader(opts, "train")

# ----------------------------------
# -----  Set global variables  -----
# ----------------------------------

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

combined_image_path = Path(opts.output_path) / Path("img") / "image_through_dae.png"
original_image_path = Path(opts.output_path) / Path("img") / "original_image_dae.png"
decoded_image_path = Path(opts.output_path) / Path("img") / "decoded_image_dae.png"

# ------------------------------
# -----  Load checkpoints  -----
# ------------------------------

checkpoints_root = Path(opts.output_path) / Path("checkpoints")

dae_checkpoint_path = None
if args.dae_checkpoint is not None:
    dae_checkpoint_path = checkpoints_root / args.dae_checkpoint
if dae_checkpoint_path is None or not dae_checkpoint_path.is_file():
    dae_checkpoint_path = checkpoints_root / "dae_latest_ckpt.pth"

# -------------------------
# -----  Load models  -----
# -------------------------

dae = DAE(
    opts.num_epochs,
    opts.data.loaders.batch_size,
    opts.dae_lr,
    opts.data.shape,
    None
)

dae_checkpoint = torch.load(dae_checkpoint_path)
dae.dae.load_state_dict(dae_checkpoint["model"])

# --------------------------------
# -----  Display DAE output  -----
# --------------------------------

if args.dae_output and not args.single_image:
    print("Commands:")
    print("\tn - pass to next image")
    print("\tq - to select the image")
    
    file = open(opts.data.files.base + opts.data.files.train, 'r')
    files = file.readlines()
    
    user_input = 'n'
    while (user_input == 'n'):
        # Read image
        index = int(np.floor(np.random.uniform(len(files))))
        image_file = files[index][:-1]
        print(image_file)
        im = PIL.Image.open(image_file)
        im_data = np.array(im).transpose(2,0,1) / 255
        shape = im_data.shape
        im_data = im_data.reshape((1, shape[0], shape[1], shape[2]))
        with torch.no_grad():
            # Resize according to model requirements
            im_tensor = torch.Tensor(im_data).to(device)
            im_tensor = torchvision.transforms.functional.resize(im_tensor, opts.data.shape[-2:])

            # Encode and decode image
            decoded_image_tensor = dae.decode(dae.encode(im_tensor))
            
        # Create a PIL images from the resulting Tensor
        original_image_array, original_image = tensor_to_PIL(im_tensor)
        decoded_image_array, decoded_image = tensor_to_PIL(decoded_image_tensor)
        
        if args.print_values:
            # Print values in console
            print(decoded_image_array)

        # Create an image combining bith original and decoded images
        both_images = PIL.Image.fromarray(
            np.hstack([original_image_array, decoded_image_array]),
            mode='RGB'
        )
        
        # Save image and decoded image together
        both_images.save(combined_image_path)
        original_image.save(original_image_path)
        decoded_image.save(decoded_image_path)

        # Get next user input
        user_input = input()

# ---------------------------------------------
# -----  Pass VAE output through the DAE  -----
# ---------------------------------------------

if args.single_image:
    # Read images
    vae_in = Path(opts.output_path) / Path("img") / "original_image_beta_vae.png"
    vae_out = Path(opts.output_path) / Path("img") / "decoded_image_beta_vae.png"
    images = [PIL.Image.open(vae_in, 'r'), PIL.Image.open(vae_out, 'r')]
    
    dae_transforms = []
    for i in range(len(images)):
        # Format the image according to model requirements
        im_data = np.array(images[i]).transpose(2,0,1) / 255
        shape = im_data.shape
        im_data = im_data.reshape((1, shape[0], shape[1], shape[2]))
        
        with torch.no_grad():
            # Resize according to model requirements
            im_tensor = torch.Tensor(im_data).to(device)
            im_tensor = torchvision.transforms.functional.resize(im_tensor, opts.data.shape[-2:])

            # Encode and decode image
            decoded_image_tensor = dae.decode(dae.encode(im_tensor))
            
        # Create a PIL images from the resulting Tensor
        original_image_array, original_image = tensor_to_PIL(im_tensor)
        decoded_image_array, decoded_image = tensor_to_PIL(decoded_image_tensor)

        # Create an array combining both original and decoded images
        both_images_array = np.hstack([original_image_array, decoded_image_array])
        dae_transforms.append(both_images_array)
        
    # Create an image combining all original and decoded images
    all_images_array = np.vstack(dae_transforms)
    all_images = PIL.Image.fromarray(all_images_array, mode='RGB')
    
    # Save all images together
    all_images.save(combined_image_path)