import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
from torchvision import transforms

from utils.utils import apply_random_mask
from dae.model import Model

from time import time
from pathlib import Path


class DAE:
    def __init__(self, num_epochs, batch_size, lr, shape, exp=None):
        # Save parameters
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.shape = shape
        self.dae = Model(shape)
        self.exp = exp
        
        # Initialize global variable
        self.global_step = 0
        
        # Initialize the model and send it to the GPU if available
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.dae.to(self.device)
        
        print("Using ...")
        print(self.device)

    def encode(self, x):
        return self.dae.encode(x)

    def decode(self, z):
        return self.dae.decode(z)

    def train(self, batches, output_path, save_n_epochs, resume=None):

        print("Training DAE...", end="", flush=True)

        # Initialize optimizer and eraser transform
        self.optimizer = optim.Adam(self.dae.parameters(), lr=self.lr)
        eraser = transforms.Lambda(apply_random_mask)
        
        # Load a checkpoint if provided
        if resume is not None:
            checkpoint = torch.load(resume)
            self.dae.load_state_dict(checkpoint["model"])
            self.optimizer.load_state_dict(checkpoint["opt"])
            self.global_step = checkpoint["step"]
            print("resumed from step " + str(self.global_step))

        for _ in range(self.num_epochs):
            print("epoch " + str(self.global_step))
            # Start step chrono
            step_start_time = time()
            for data in batches:
                # Apply eraser on batch images
                erased_data = data.detach().clone()
                for i, example in enumerate(erased_data):
                    erased_data[i] = eraser.lambd(erased_data[i])
                erased_data = erased_data.to(self.device)

                # Pass the image through the beta-VAE
                out = self.dae(erased_data)
                data = data.to(self.device)

                # Calculate loss
                loss = torch.pow(data - out, 2).mean()
                
                # Backpropagate the loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # Stop step chrono and increment step number
            step_time = time() - step_start_time
            self.global_step += 1
            
            # Log training step in comet ml
            if self.exp is not None:
                self.exp.log_metric("dae_train_loss", loss, step=self.global_step)
                self.exp.log_metric("Step-time", step_time, step=self.global_step)

            # Save optimizer
            self.save(output_path, save_n_epochs)

        # End of training
        print("DONE")

    def save(self, output_path, save_n_epochs):
        # Initialize paths and saving directory
        save_dir = Path(output_path) / Path("checkpoints")
        save_dir.mkdir(exist_ok=True)
        save_path = save_dir / "dae_latest_ckpt.pth"

        # Construct relevant state dicts / optims:
        save_dict = {
            "model": self.dae.state_dict(),
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
                save_dir / f"dae_epoch_{self.global_step}_ckpt.pth"
            )
            print("Model saved in " + str(save_path))