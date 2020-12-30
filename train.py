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


def main():
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
        "--no-comet", action="store_true", help="launch comet exp or not"
    )
    parser.add_argument(
        "--comet-tags", type=str, default=None, help="tags for comet exp"
    )
    parser.add_argument(
        "--dae-checkpoint", type=str, default=None, help="dae checkpoint from which to start the training"
    )
    parser.add_argument(
        "--vae-checkpoint", type=str, default=None, help="vae checkpoint from which to start the training"
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
            exp = Experiment(
                api_key=opts.comet.api_key,
                project_name=opts.comet.project_name,
                workspace=opts.comet.workspace
            )
            exp.log_asset_folder(
                str(Path(__file__).parent / "beta_vae"),
                recursive=True,
                log_file_name=True,
            )
            exp.log_asset_folder(
                str(Path(__file__).parent / "config"),
                recursive=True,
                log_file_name=True,
            )
            exp.log_asset_folder(
                str(Path(__file__).parent / "dae"),
                recursive=True,
                log_file_name=True,
            )
            exp.log_asset_folder(
                str(Path(__file__).parent / "data"),
                recursive=True,
                log_file_name=True,
            )
            exp.log_asset_folder(
                str(Path(__file__).parent / "utils"),
                recursive=True,
                log_file_name=True,
            )
            exp.log_asset(str(Path(__file__).parent / "README.md"))
            exp.log_asset(str(Path(__file__)))

        # log tags
        tags = set()
        tags.add(str(opts.module))
        if args.comet_tags:
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

    # ------------------------------
    # -----  Load checkpoints  -----
    # ------------------------------

    checkpoints_root = Path(opts.output_path) / Path("checkpoints")
    
    dae_checkpoint_path = None
    if args.dae_checkpoint is not None:
        dae_checkpoint_path = checkpoints_root / args.dae_checkpoint
    if dae_checkpoint_path is None or not dae_checkpoint_path.is_file():
        if opts.module == "beta_vae":
            dae_checkpoint_path = checkpoints_root / "dae_latest_ckpt.pth"

    vae_checkpoint_path = None
    if args.vae_checkpoint is not None:
        vae_checkpoint_path = checkpoints_root / args.vae_checkpoint

    # -------------------
    # -----  Train  -----
    # -------------------

    loader = get_loader(opts, "train")

    shape = opts.data.shape

    if opts.module == "dae":
        module = DAE(
            opts.num_epochs,
            opts.data.loaders.batch_size,
            opts.dae_lr,
            opts.save_iter,
            opts.data.shape,
            exp,
        )
        module.train(
            loader,
            opts.output_path,
            opts.save_n_epochs,
            dae_checkpoint_path
        )

    elif opts.module == "beta_vae":
        module = BetaVAE(
            opts.data.n_obs,
            opts.num_epochs,
            opts.data.loaders.batch_size,
            opts.betavae_lr,
            opts.beta,
            opts.latent_dim,
            opts.save_iter,
            opts.data.shape,
            exp,
        )
        dae = DAE(
            opts.num_epochs,
            opts.data.loaders.batch_size,
            opts.dae_lr,
            opts.save_iter,
            opts.data.shape,
            None,
        )
        dae_checkpoint = torch.load(dae_checkpoint_path)
        dae.dae.load_state_dict(dae_checkpoint["model"])        
        module.train(
            loader, 
            dae.dae, 
            opts.output_path, 
            opts.save_n_epochs,
            vae_checkpoint_path)

    # -----------------------------
    # -----  End of training  -----
    # -----------------------------

    print("Done training")
    # kill_job(opts.jobID)


if __name__ == "__main__":

    main()
