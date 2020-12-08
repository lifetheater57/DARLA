import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image

from beta_vae.model import Model
from beta_vae.visualize import *
from time import time


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

        # TODO: make 32 (latent_dim) configurable
        self.vae = Model(shape, 32)

    def encode(self, x):
        return self.vae.encode(x)

    def decode(self, z):
        return self.vae.decode(z)

    def represent(self, x):
        return self.vae.represent(x)

    def train(self, batches, dae):
        print("Training ß-VAE...", end="", flush=True)

        def KL(mu, log_var):
            kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            kl /= mu.size(0) * self.n_obs
            return kl

        optimizer = optim.Adam(self.vae.parameters(), lr=self.lr)

        for epoch in range(self.num_epochs):
            step_start_time = time()
            for data in batches:

                out, mu, log_var = self.vae(data)

                # calculate loss and update network
                x_bar = dae.decode(dae.encode(data))
                x_hat_bar = dae.encode(dae.encode(out))

                reconstruction_loss = torch.pow(x_bar - x_hat_bar, 2).mean()
                loss = reconstruction_loss + (self.beta * KL(mu, log_var))

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
                self.exp.log_metrics(losses, prefix="vae_train", step=self.global_step)
                self.exp.log_metric("Step-time vae", step_time, step=self.global_step)

        print("DONE")
