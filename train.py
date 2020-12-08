from pathlib import Path
from time import time, sleep
import os
import yaml
from addict import Dict
import argparse
from comet_ml import Experiment, ExistingExperiment
from data.data import *
from utils.utils import env_to_path, copy_run_files, flatten_opts
from dae.dae import DAE
from beta_vae.beta_vae import BetaVAE


def main(opts):
    """
    python train.py --config path_to_config.yaml 

    """

    # -----------------------------
    # -----  Parse arguments  -----
    # -----------------------------

    parser = argparse.ArgumentParser(description="train model")
    parser.add_argument(
        "--config",
        type=str,
        default="./config/defaults.yaml",
        help="path to config file",
    )
    parser.add_argument(
        "--no_comet", type=bool, action="store_true", help="launch comet exp or not"
    )
    parser.add_argument(
        "--comet_tags", type=str, default=None, help="tags for comet exp"
    )
    args = parser.parse_args()
    # -----------------------
    # -----  Load opts  -----
    # -----------------------

    with open(args.config, "r") as f:
        opts = yaml.safe_load(f)

    opts = Dict(opts)
    opts.jobID = os.environ.get("SLURM_JOBID")
    opts.output_path = str(env_to_path(opts.output_path))
    print("Config output_path:", opts.output_path)

    exp = comet_previous_id = None
    # create output_path if it doesn't exist
    Path(opts.output_path).mkdir(parents=True, exist_ok=True)

    # Copy the opts's sbatch_file to output_path
    copy_run_files(opts)

    if not args.no_comet:
        # ----------------------------------
        # -----  Set Comet Experiment  -----
        # ----------------------------------
        if exp is None:
            # Create new experiment
            print("Starting new experiment")
            exp = Experiment(project_name="DARLA", **comet_kwargs)
            exp.log_asset_folder(
                str(Path(__file__).parent / "DARLA"),
                recursive=True,
                log_file_name=True,
            )
            exp.log_asset(str(Path(__file__)))

        # log tags
        if args.comet_tags:
            tags = set()
            if args.comet_tags:
                tags.update(args.comet_tags)
            opts.comet.tags = list(tags)
            print("Logging to comet.ml with tags", opts.comet.tags)
            exp.add_tags(opts.comet.tags)

        # Log all opts
        exp.log_parameters(flatten_opts(opts))

        # allow some time for comet to get its url
        sleep(1)

        # Save comet exp url
        url_path = Path(opts.output_path) / "comet_url.txt"
        with open(url_path, "w") as f:
            f.write(exp.url)

        # Save config file
        opts_path = Path(opts.output_path) / "opts.yaml"
        with (opts_path).open("w") as f:
            yaml.safe_dump(opts.to_dict(), f)

    print("Running model in", opts.output_path)

    # -------------------
    # -----  Train  -----
    # -------------------

    loader = get_loader(opts, "train")

    shape = opts.data.shape

    if opts.module == "dae":
        module = DAE(
            opts.data.n_obs,
            opts.num_epochs,
            opts.data.loaders.batch_size,
            opts.dae_lr,
            opts.save_iter,
            opts.data.shape,
            exp,
        )
    elif opts.module == "beta_vae":
        module = BetaVAE(
            opts.data.n_obs,
            num_epochs,
            opts.data.loaders.batch_size,
            opts.betavae_lr,
            opts.beta,
            opts.save_iter,
            opts.data.shape,
        )
        # TODO  Modify betaVAE file

    module.train(loader)

    # -----------------------------
    # -----  End of training  -----
    # -----------------------------

    print("Done training")
    # kill_job(opts.jobID)


if __name__ == "__main__":

    main()

"""import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

from dae.dae import DAE
from beta_vae.beta_vae import BetaVAE
from history import History

# hyperparameters
num_epochs = 100
batch_size = 128
lr = 1e-4
beta = 4
save_iter = 20

shape = (28, 28)
n_obs = 3

# create DAE and ß-VAE and their training history
dae = DAE(n_obs, num_epochs, batch_size, 1e-3, save_iter, shape)
beta_vae = BetaVAE(n_obs, num_epochs, batch_size, 1e-4, beta, save_iter, shape)
history = History()

# fill autoencoder training history with examples
print('Filling history...', end='', flush=True)

transformation = transforms.Compose([
    transforms.ColorJitter(),
    transforms.ToTensor()
])

dataset = MNIST('data', transform=transformation)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

for data in dataloader:
    img, _ = data
    img = img.view(img.size(0), -1).numpy().tolist()
    history.store(img)
print('DONE')

# train DAE
dae.train(history)

# train ß-VAE
beta_vae.train(history, dae)
"""
