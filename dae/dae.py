import torch
from comet_ml import Experiment
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
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

        optimizer = optim.Adam(self.dae.parameters(), lr=self.lr)

        for epoch in range(self.num_epochs):
            print("epoch " + str(epoch))
            step_start_time = time()
            for data in batches:
                out = self.dae(data)

                # calculate loss and update network
                loss = torch.pow(data - out, 2).mean()
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
                self.exp.log_metric(loss, prefix="dae_train", step=self.global_step)
                self.exp.log_metric("Step-time", step_time, step=self.global_step)

            self.save(optimizer, output_path, save_n_epochs)

        print("DONE")

    def save(self, optimizer, output_path, save_n_epochs):
        save_dir = Path(output_path) / Path("checkpoints")
        save_dir.mkdir(exist_ok=True)
        save_path = save_dir / "dae_latest_ckpt.pth"

        # Construct relevant state dicts / optims:
        # Save at least G
        save_dict = {
            "G": self.dae.state_dict(),
            "opt": optimizer.state_dict(),
            "step": self.global_step,
        }
        if self.global_step % save_n_epochs == 0:
            torch.save(save_dict, save_dir / f"dae_epoch_{self.global_step}_ckpt.pth")

        torch.save(save_dict, save_path)
        print("saved model in " + save_path)


def get_random_mask(H, W):
    """get binary image of a mask random box area 

    Args:
        H ([type]): [description]
        W ([type]): [description]
    """
    rng = np.random.default_rng()
    sample_heights = sorted(rng.choice(H, size=2, replace=False))
    sample_widths = sorted(rng.choice(W, size=2, replace=False))
    box = Image.new(
        "RGB",
        (sample_width[1] - sample_width[0], sample_height[1] - sample_height[0]),
        (0, 0, 0),
    )
    mask = Image.new("RGB", (W, H), (255, 255, 255))
    mask.paste(box, (sample_width[0], sample_height[0]))
    return mask
