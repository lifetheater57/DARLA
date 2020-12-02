import numpy as np
import os
from PIL import Image

# ---------------------------------------------
# ----- example for making train.txt file -----
# ---------------------------------------------

# save .npy as .png images

path = "/miniscratch/tengmeli/duckietown/ds_pt2.npy"
data = np.load(path)

for i in range(len(data)):
    num = str(i)
    for j in range(4 - len(str(i))):
        num = "0" + num
    Image.fromarray(data[i]).save(
        os.path.join("/miniscratch/tengmeli/duckietown/ds_imgs/", "pt2_" + num + ".png")
    )

# write paths in txt
ims = os.listdir("/miniscratch/tengmeli/duckietown/ds_imgs")
paths = [os.path.join("/miniscratch/tengmeli/duckietown/ds_imgs", i) for i in ims]

with open("train.txt", "w") as f:
    for item in paths:
        f.write("%s\n" % item)
