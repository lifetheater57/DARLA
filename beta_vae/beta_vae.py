import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image

from beta_vae.model import Model
from time import time
from pathlib import Path

from utils.latent_space import traversals


class BetaVAE:
    def __init__(
        self, n_obs, num_epochs, batch_size, lr, beta, save_iter, shape, exp=None
    ):
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.beta = beta
        self.save_iter = save_iter
        self.shape = shape  # c*h*w

        self.n_obs = n_obs
        self.exp = exp
        self.global_step = 0
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # TODO: make 32 (latent_dim) configurable
        self.vae = Model(shape, 32)

        self.vae.to(self.device)
        print("Using ...")
        print(self.device)

    def encode(self, x):
        return self.vae.encode(x)

    def decode(self, z):
        return self.vae.decode(z)

    def represent(self, x):
        return self.vae.represent(x)

    def train(self, batches, dae, output_path, save_n_epochs):
        print("Training ß-VAE...", end="", flush=True)

        def KL(mu, log_var):
            kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            kl /= mu.size(0) * self.n_obs
            return kl

        optimizer = optim.Adam(self.vae.parameters(), lr=self.lr)

        for epoch in range(self.num_epochs):
            step_start_time = time()
            for data in batches:
                data = data.to(self.device)
                out, mu, log_var = self.vae(data)

                # calculate loss and update network
                x_bar = dae.decode(dae.encode(data))
                x_hat_bar = dae.decode(dae.encode(out))

                reconstruction_loss = torch.pow(x_bar - x_hat_bar, 2).mean()
                loss = reconstruction_loss + (self.beta * KL(mu, log_var)).to(
                    self.device
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # -----------------
            # -----  Log  -----
            # -----------------
            step_time = time() - step_start_time
            self.global_step += 1
            # log in comet ml
            if self.exp is not None:
                self.exp.log_metric("vae_train_loss", loss, step=self.global_step)
                self.exp.log_metric("Step-time vae", step_time, step=self.global_step)
            self.save(optimizer, output_path, save_n_epochs)
        print("DONE")

    def visualize_traversal(self, x):
        # build sample from image x
        latent_sample = self.encode(x)
        decoder = self.vae.decoder
        shape = self.shape

        # (decoder, shape, dimensions, bounds, num_samples, state)

    def save(self, optimizer, output_path, save_n_epochs):
        save_dir = Path(output_path) / Path("checkpoints")
        save_dir.mkdir(exist_ok=True)
        save_path = save_dir / "beta_vae_latest_ckpt.pth"

        # Construct relevant state dicts / optims:

        save_dict = {
            "model": self.vae.state_dict(),
            "opt": optimizer.state_dict(),
            "step": self.global_step,
        }
        if self.global_step % save_n_epochs == 0:
            torch.save(
                save_dict, save_dir / f"beta_vae_epoch_{self.global_step}_ckpt.pth"
            )
            print("saved model in " + str(save_path))

        torch.save(save_dict, save_path)
        print("saved model in " + str(save_path))
