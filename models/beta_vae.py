import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.utils import save_image

import numpy as np
from visdom import Visdom

viz = Visdom()

title = 'ß-VAE Loss by Epoch'
win = None

def update_viz(epoch, loss):
    global win, title

    if win is None:
        title = title

        win = viz.line(
            X=np.array([epoch]),
            Y=np.array([loss]),
            win=title,
            opts=dict(
                title=title,
                fillarea=True
            )
        )
    else:
        viz.line(
            X=np.array([epoch]),
            Y=np.array([loss]),
            win=win,
            update='append'
        )

class Model(nn.Module):
    def __init__(self, n_obs):
        super(Model, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(n_obs, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU()
        )

        self.mu = nn.Sequential(
            nn.Linear(64, 10)
        )

        self.log_var = nn.Sequential(
            nn.Linear(64, 10)
        )

        self.decoder = nn.Sequential(
            nn.Linear(10, 64),
            nn.ELU(),
            nn.Linear(64, 128),
            nn.ELU(),
            nn.Linear(128, n_obs),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        mu = self.mu(x)
        log_var = self.log_var(x)

        z = mu + torch.mul(torch.exp(log_var / 2), torch.randn_like(log_var))
        x_hat = self.decoder(z)

        return x_hat, mu, log_var

    def encode(self, x):
        x = self.encoder(x)
        mu = self.mu(x)
        log_var = self.log_var(x)
        z = mu + torch.mul(torch.exp(log_var / 2), torch.randn_like(log_var))
        return z

    def decode(self, z):
        return self.decoder(z)

class BetaVAE():
    def __init__(self, n_obs, num_epochs, batch_size, lr, beta):
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.beta = beta

        self.n_obs = n_obs

        self.vae = Model(n_obs)

    def encode(self, x):
        return self.vae.encode(x)

    def decode(self, z):
        return self.vae.decode(z)

    def train(self, history, dae):
        print('Training ß-VAE...', end='', flush=True)

        def KL(mu, log_var):
            kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            kl /= mu.size(0) * self.n_obs
            return kl

        optimizer = optim.Adam(self.vae.parameters(), lr=self.lr)

        for epoch in range(self.num_epochs):

            minibatches = history.get_minibatches(self.batch_size, self.num_epochs)
            for data in minibatches:

                out, mu, log_var = self.vae(data)

                # calculate loss and update network
                loss = F.mse_loss(dae.encode(data), dae.encode(out)) + (self.beta * KL(mu, log_var))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if epoch == 0 or epoch % 20 == 19:
                pic = out.data.view(out.size(0), 1, 28, 28)
                save_image(pic, 'img/betaVae_' + str(epoch+1) + '_epochs.png')

            # plot loss
            update_viz(epoch, loss.item())

        print('DONE')
