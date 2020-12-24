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
            exp = Experiment(project_name="DARLA")
            exp.log_asset_folder(
                str(Path(__file__).parent / "DARLA"),
                recursive=True,
                log_file_name=True,
            )
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
        # module.dae.cuda()
        # if opts.resume != "":
        #    checkpoint = torch.load(
        #        Path(opts.output_path) / Path("checkpoints") / opts.resume
        #     )  # "dae_latest_ckpt.pth")

        #      module.dae.load_state_dict(checkpoint["model"])
        #      module.dae.
        module.train(
            loader,
            opts.output_path,
            opts.save_n_epochs,
            #        resume=Path("/miniscratch/tengmeli/DARLA_small_bs128")
            #        / Path("checkpoints")
            #       / "dae_latest_ckpt.pth",
        )

    elif opts.module == "beta_vae":
        module = BetaVAE(
            opts.data.n_obs,
            opts.num_epochs,
            opts.data.loaders.batch_size,
            opts.betavae_lr,
            opts.beta,
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
        checkpoint = torch.load(
            Path(opts.output_path) / Path("checkpoints") / "dae_latest_ckpt.pth"
        )
        dae.dae.load_state_dict(checkpoint["model"])
        # module.vae.cuda()
        module.train(loader, dae.dae, opts.output_path, opts.save_n_epochs)

    # -----------------------------
    # -----  End of training  -----
    # -----------------------------

    print("Done training")
    # kill_job(opts.jobID)


if __name__ == "__main__":

    main()