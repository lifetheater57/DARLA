from pathlib import Path
import os
import datetime
import yaml
from addict import Dict
import argparse
from utils.utils import env_to_path, tensor_to_PIL
from utils import latent_space
from data.data import *
from beta_vae.beta_vae import BetaVAE
import torch
import numpy as np
import PIL
import torchvision
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
parser.add_argument(
    "--vae-checkpoint", type=str, default=None, help="vae model checkpoint to use"
)
parser.add_argument(
    "--bounds-checkpoint", type=str, default=None, help="vae latent space bounds checkpoint to use"
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

# ------------------------------
# -----  Load checkpoints  -----
# ------------------------------

checkpoints_root = Path(opts.output_path) / Path("checkpoints")
    
vae_checkpoint_path = None
if args.vae_checkpoint is not None:
    vae_checkpoint_path = checkpoints_root / args.vae_checkpoint
if vae_checkpoint_path is None or not vae_checkpoint_path.is_file():
    vae_checkpoint_path = checkpoints_root / "beta_vae_latest_ckpt.pth"

bounds_path = None
if args.bounds_checkpoint is not None:
    bounds_path = checkpoints_root / args.vae_checkpoint
if bounds_path is None or not bounds_path.is_file():
    bounds_path = checkpoints_root / "bounds_latest_ckpt.npy"

# -------------------------
# -----  Load models  -----
# -------------------------

if args.select_state or args.generate_bounds or args.generate_traversals:
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

    vae_checkpoint = torch.load(vae_checkpoint_path)
    beta_vae.vae.load_state_dict(vae_checkpoint["model"])

# -----------------------------
# -----  Generate bounds  -----
# -----------------------------

bounds = None

if Path(bounds_path).is_file():
    bounds = np.load(bounds_path)

if args.generate_bounds or bounds is None:
    if bounds is None:
        print("Automatically generating latent space dimension bounds.")
    bounds = latent_space.get_bounds(loader, beta_vae.vae)

    np.save(bounds_path, bounds)
    print("Latent space dimension bounds saved at: " + str(bounds_path))
    
# --------------------------
# -----  Manage state  -----
# --------------------------

# Initialize the state
scale = bounds[:, 1] - bounds[:, 0]
state = np.random.normal(scale=scale, size=opts.latent_dim)

if args.state is not None:
    # Use the state given in argument
    state = args.state
    print("Using state: " + str(state))
    #if len(args.state) == opts.latent_dim:
    #else:
    #    print("Need a state with " + str(opts.latent_dim) + "dimensions.")
    #    print("Keeping random state.")

# Send the state to the device to use
state = torch.Tensor(state).to(device)

if args.select_state:
    # Print instructions
    print("Commands:")
    print("\tn - pass to next state")
    print("\ts - to select the state")
    print("\tq - to select the state")
    
    # Open image file list
    file = open(opts.data.files.base + opts.data.files.train, 'r')
    files = file.readlines()

    # Set the path of the images to save
    combined_image_path = Path(opts.output_path) / Path("img") / "image_through_beta_vae.png"
    original_image_path = Path(opts.output_path) / Path("img") / "original_image_beta_vae.png"
    decoded_image_path = Path(opts.output_path) / Path("img") / "decoded_image_beta_vae.png"
    
    # Initialize the loop
    user_input = 'n'
    while (user_input == 'n'):
        # Read image
        index = int(np.floor(np.random.uniform(len(files))))
        image_file = files[index][:-1]
        im = PIL.Image.open(image_file)     
        print("Image file:", image_file)
        
        # Format the image according to model requirements
        im_data = np.array(im).transpose(2,0,1) / 255
        shape = im_data.shape
        im_data = im_data.reshape((1, shape[0], shape[1], shape[2]))
        im_data = np.vstack([im_data] * opts.data.loaders.batch_size)
        
        with torch.no_grad():
            # Resize according to model requirements
            im_tensor = torch.Tensor(im_data).to(device)
            im_tensor = torchvision.transforms.functional.resize(im_tensor, opts.data.shape[-2:])

            # Compute encoded image
            temp_state = beta_vae.encode(im_tensor)
            temp_state_str = str(temp_state).replace("tensor([[", '')
            temp_state_str = temp_state_str.replace("]], device='cuda:0')", '')
            temp_state_str = str.join(' ', str.split(temp_state_str, ','))
            print("state: " + temp_state_str)

            # Decode state
            decoded_image_tensor = beta_vae.decode(temp_state)
            
        # Create a PIL images from the resulting Tensor
        original_image_array, original_image = tensor_to_PIL(im_tensor)
        decoded_image_array, decoded_image = tensor_to_PIL(decoded_image_tensor)

        # Create an image combining both original and decoded images
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

        # Save the stat as the current state to use if required
        if user_input == 's':
            state = temp_state
            user_input = 'n'

# ---------------------------------
# -----  Generate traversals  -----
# ---------------------------------

if args.generate_traversals:
    # Get dimension list from arguments
    dimensions = args.dimensions

    if dimensions is None:
        # Select all dimensions of the latent space
        dimensions = range(opts.latent_dim)

    # Set the path of the traversals figure's png
    traversal_path = Path(opts.output_path) / Path("img") / "traversals.png"

    # Generate the traversals figure
    latent_space.traversals(
        beta_vae.vae,
        opts.data.shape,
        dimensions,
        bounds,
        8,
        state,
        traversal_path
    )