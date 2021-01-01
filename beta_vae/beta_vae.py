import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
from torchvision import transforms

from beta_vae.model import Model
from utils.utils import apply_random_mask

from time import time
from pathlib import Path


class BetaVAE:
    def __init__(
        self, num_epochs, batch_size, lr, beta, latent_dim, shape, exp=None
    ):
        # Save parameters
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.beta = beta
        self.latent_dim = latent_dim
        self.shape = shape  # c*h*w
        self.exp = exp

        # Initialize global variable
        self.global_step = 0
        
        # Initialize the model and send it to the GPU if available
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.vae = Model(shape, self.latent_dim)
        self.vae.to(self.device)

        print("Using ...")
        print(self.device)

    def encode(self, x):
        return self.vae.encode(x)

    def decode(self, z):
        return self.vae.decode(z)

    def train(self, batches, dae, output_path, save_n_epochs, resume=None):
        print("Training ß-VAE...", end="", flush=True)

        def KL(mu, log_var):
            kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            kl /= mu.size(0) * self.shape[0]
            return kl

        # Initialize the optimizer
        self.optimizer = optim.Adam(self.vae.parameters(), lr=self.lr)
        
        # Create an eraser to a apply a random occlusion window if required
        if dae is None:
            eraser = transforms.Lambda(apply_random_mask)
        
        # Load a checkpoint if provided
        if resume is not None:
            checkpoint = torch.load(resume)
            self.vae.load_state_dict(checkpoint["model"])
            self.optimizer.load_state_dict(checkpoint["opt"])
            self.global_step = checkpoint["step"]
            print("resumed from step " + str(self.global_step))

        # Train over the required number of epoch
        for _ in range(self.num_epochs):
            print("epoch " + str(self.global_step))
            # Start step chrono
            step_start_time = time()
            for data in batches:
                if dae is None:
                    # Apply eraser on batch images
                    erased_data = data.detach().clone()
                    for i, example in enumerate(erased_data):
                        erased_data[i] = eraser.lambd(erased_data[i])
                    erased_data = erased_data.to(self.device)
                else:
                    # Keep the data unchanged
                    erased_data = data.to(self.device)
                
                # Pass the image through the beta-VAE
                out, mu, log_var = self.vae(erased_data)
                data = data.to(self.device)

                # Get values to compute the loss
                if dae is None:
                    x_bar = data
                    x_hat_bar = out
                else:
                    x_bar = dae.decode(dae.encode(data))
                    x_hat_bar = dae.decode(dae.encode(out))

                # Calculate the loss value
                reconstruction_loss = torch.pow(x_bar - x_hat_bar, 2).mean()
                kl_loss = (self.beta * KL(mu, log_var)).to(self.device)
                loss = reconstruction_loss + kl_loss 

                # Backpropagate the loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # Stop step chrono and increment step number
            step_time = time() - step_start_time
            self.global_step += 1
            
            # Log training step in comet ml
            if self.exp is not None:
                self.exp.log_metric("vae_train_loss", loss, step=self.global_step)
                self.exp.log_metric("Step-time vae", step_time, step=self.global_step)
            
            # Save optimizer
            self.save(output_path, save_n_epochs)
        
        # End of training
        print("DONE")

    def save(self, output_path, save_n_epochs):
        # Initialize paths and saving directory
        save_dir = Path(output_path) / Path("checkpoints")
        save_dir.mkdir(exist_ok=True)
        save_path = save_dir / "beta_vae_latest_ckpt.pth"

        # Construct relevant state dicts / optims:
        save_dict = {
            "model": self.vae.state_dict(),
            "opt": self.optimizer.state_dict(),
            "step": self.global_step,
        }

        # Save as the latest checkpoint
        torch.save(save_dict, save_path)
        print("Model saved in " + str(save_path))

        # Save as a numbered checkpoint if required
        if self.global_step % save_n_epochs == 0:
            torch.save(
                save_dict, 
                save_dir / f"beta_vae_epoch_{self.global_step}_ckpt.pth"
            )
            print("Model saved in " + str(save_path))