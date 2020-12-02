import os
import sys
from pathlib import Path
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))

from PIL import Image
import json
import yaml
from data.data import *
from dae.dae import DAE
from addict import Dict

if __name__ == "__main__":
    with open("../config/defaults.yaml", "r") as f:
        opts = yaml.safe_load(f)

    opts = Dict(opts)
    loader = get_loader(opts, "train")
    batch = next(iter(loader))
    _, h, w = batch[0].shape

    num_epochs = opts.num_epochs
    batch_size = opts.data.loaders.batch_size
    lr = opts.lr
    beta = 4
    save_iter = 20

    shape = (h, w)
    n_obs = 3

    # create DAE and ß-VAE and their training history
    dae = DAE(n_obs, num_epochs, batch_size, 1e-3, save_iter, shape)
    out = dae.dae(batch)
    print(out)
