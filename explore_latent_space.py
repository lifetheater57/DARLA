from pathlib import Path
import os
import datetime
import yaml
from addict import Dict
import argparse
from utils.utils import env_to_path
from utils import latent_space
from data.data import *
from beta_vae.beta_vae import BetaVAE
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
    "--select-state", action="store_true", help="generate the latent space dimension bounds"
)
parser.add_argument(
    "--state",
    metavar='f', 
    type=float,
    nargs='+',
    default=None,
    help="state to use if --select-state is not given or if no state have been selected"
)
parser.add_argument(
    "--generate-bounds", action="store_true", help="generate the latent space dimension bounds"
)
parser.add_argument(
    "--generate-traversals", action="store_true", help="generate a traversals of the latent space"
)
parser.add_argument(
    "--dimensions", 
    metavar='N', 
    type=int,
    nargs='+',
    default=None,
    help="dimentions to traverse"
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

# ------------------------
# -----  Load model  -----
# ------------------------

beta_vae = BetaVAE(
    opts.num_epochs,
    opts.data.loaders.batch_size,
    opts.betavae_lr,
    opts.beta,
    opts.latent_dim,
    opts.save_iter,
    opts.data.shape,
    None
)

checkpoint = torch.load(
    Path(opts.output_path) / Path("checkpoints") / "beta_vae_latest_ckpt.pth"
)

beta_vae.vae.load_state_dict(checkpoint["model"])

# -----------------------------
# -----  Generate bounds  -----
# -----------------------------

bounds = None

bounds_path = Path(opts.output_path) / Path("checkpoints") / "bounds_latest_ckpt.npy"

if Path(bounds_path).is_file():
    bounds = np.load(bounds_path)

if args.generate_bounds or bounds is None:
    if bounds is None:
        print("Automatically generating latent space dimension bounds.")
    bounds = latent_space.get_bounds(loader, beta_vae.vae)

    np.save(bounds_path, bounds)
    print("Latent space dimension bounds saved at: " + str(bounds_path))
    
    #bounds.save(Path(opts.output_path) / Path("checkpoints") / "bounds_" + datetime.now() + ".npy")

# --------------------------
# -----  Manage state  -----
# --------------------------

# Initialize the state
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
scale = bounds[:, 1] - bounds[:, 0]
state = np.random.normal(scale=scale, size=opts.latent_dim)

if args.state is not None:
    state = args.state
    print("Using state: " + str(state))
    #if len(args.state) == opts.latent_dim:
    #else:
    #    print("Need a state with " + str(opts.latent_dim) + "dimensions.")
    #    print("Keeping random state.")

state = torch.Tensor(state).to(device)

if args.select_state:
    print("Commands:")
    print("\tn - pass to next state")
    print("\ts - to select the state")
    print("\tq - to select the state")
    
    file = open(opts.data.files.base + opts.data.files.train, 'r')
    files = file.readlines()

    state_path = Path(opts.output_path) / Path("img") / "decoded_state.png"
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
            im_tensor = torch.Tensor(im_data).to(device)
            im_tensor = torchvision.transforms.functional.resize(im_tensor, opts.data.shape[-2:])

            # Compute encoded image
            temp_state = beta_vae.encode(im_tensor)
            temp_state_str = str(temp_state).replace("tensor([[", '')
            temp_state_str = temp_state_str.replace("]], device='cuda:0')", '')
            temp_state_str = str.join(' ', str.split(temp_state_str, ','))
            print("state: " + temp_state_str)

            # Decode state
            decoded_image = beta_vae.decode(temp_state)
            
        # Transpose into a shape usable by matplotlib
        original_image = im_tensor.cpu().numpy().squeeze(axis=0)
        original_image = original_image.transpose(1, 2, 0)
        
        decoded_image = decoded_image.cpu().numpy().squeeze(axis=0)
        decoded_image = decoded_image.transpose(1, 2, 0)

        # Create a figure with original image and decoded image
        figure = np.hstack([original_image, decoded_image])
        
        # Save image and decoded image together
        plt.figure(figsize=(24, 9))
        plt.imshow(figure)
        plt.savefig(state_path)

        # Get next user input
        user_input = input()

        if user_input == 's':
            state = temp_state
            user_input = 'n'

# ---------------------------------
# -----  Generate traversals  -----
# ---------------------------------

if args.generate_traversals:
    dimensions = args.dimensions

    if dimensions is None:
        dimensions = range(opts.latent_dim)

    traversal_path = Path(opts.output_path) / Path("img") / "traversals.png"

    latent_space.traversals(
        beta_vae.vae,
        opts.data.shape,
        dimensions,
        bounds,
        8,
        state,
        traversal_path
    )