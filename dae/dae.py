import torch
from comet_ml import Experiment
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
from torchvision import transforms
from PIL import Image
from dae.model import Model
from time import time
from pathlib import Path


class DAE:
    def __init__(self, num_epochs, batch_size, lr, save_iter, shape, exp=None):
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.save_iter = save_iter
        self.shape = shape
        self.global_step = 0
        self.dae = Model(shape)
        self.exp = exp
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dae.to(self.device)
        print("Using ...")
        print(self.device)

    def encode(self, x):
        return self.dae.encode(x)

    def decode(self, z):
        return self.dae.decode(z)

    def update_net(self, batch, verbose=0):
        """
        Args:
            batch (dict): dictionnary of domain batches
        """
        zero_grad(self.dae)
        g_loss = self.get_g_loss(multi_domain_batch, verbose)
        g_loss.backward()
        self.g_opt_step()
        self.log_losses(model_to_update="G", mode="train")

    def train(self, batches, output_path, save_n_epochs):

        print("Training DAE...", end="", flush=True)

        # Initialize optimizer and eraser transform
        self.optimizer = optim.Adam(self.dae.parameters(), lr=self.lr)
        eraser = transforms.Lambda(apply_random_mask)

        for epoch in range(self.num_epochs):
            print("epoch " + str(epoch))
            step_start_time = time()
            for data in batches:
                # Apply eraser on batch images
                erased_data = data.detach().clone()
                for i, example in enumerate(erased_data):
                    erased_data[i] = eraser.lambd(erased_data[i])
                erased_data = erased_data.to(self.device)

                out = self.dae(erased_data)
                data = data.to(self.device)

                # calculate loss and update network
                loss = torch.pow(data - out, 2).mean()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # -----------------
            # -----  Log  -----
            # -----------------
            step_time = time() - step_start_time
            self.global_step += 1
            # log in comet ml
            if self.exp is not None:
                self.exp.log_metric("dae_train_loss", loss, step=self.global_step)
                self.exp.log_metric("Step-time", step_time, step=self.global_step)

            self.save(self.optimizer, output_path, save_n_epochs)

        print("DONE")

    def save(self, optimizer, output_path, save_n_epochs):
        save_dir = Path(output_path) / Path("checkpoints")
        save_dir.mkdir(exist_ok=True)
        save_path = save_dir / "dae_latest_ckpt.pth"

        # Construct relevant state dicts / optims:

        save_dict = {
            "model": self.dae.state_dict(),
            "opt": self.optimizer.state_dict(),
            "step": self.global_step,
        }
        if self.global_step % save_n_epochs == 0:
            torch.save(save_dict, save_dir / f"dae_epoch_{self.global_step}_ckpt.pth")

        torch.save(save_dict, save_path)
        print("saved model in " + str(save_path))


def apply_random_mask(img):
    """Blank a rectangular region of random dimensions in the image.

    Args:
        img (tensor): The image on which to apply the mask.
    """

    img_c, img_h, img_w = img.shape[-3], img.shape[-2], img.shape[-1]
    
    h_values = torch.empty(2).uniform_(0, img_h)
    w_values = torch.empty(2).uniform_(0, img_w)

    x = h_values.min().type(torch.IntTensor)
    y = w_values.min().type(torch.IntTensor)

    h = torch.abs(h_values[1] - h_values[0]).type(torch.IntTensor)
    w = torch.abs(w_values[1] - w_values[0]).type(torch.IntTensor)

    return transforms.functional.erase(img, x, y, h, w, 0)
