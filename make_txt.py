import numpy as np
import os
from PIL import Image
import sys
from pathlib import Path

# ---------------------------------
# ----- making train.txt file -----
# ---------------------------------

# Setting source and target paths
if len(sys.argv) > 2:
    target_path = Path(sys.argv[1])
    source_path = Path(sys.argv[2])
elif len(sys.argv) > 1:
    target_path = Path(sys.argv[1])
    source_path = None
else:
    target_path = Path("../../duckietown_rl/datasets")
    source_path = None
    
if source_path is not None and source_path.match("*.npy"):
    # Create output_path if it doesn't exist
    target_path.mkdir(parents=True, exist_ok=True)

    # Load the data
    data = np.load(source_path)

    # Save the image in individual png files
    ds_ID = str(source_path).split("/")[-1].replace(".npy", "")
    for i in range(len(data)):
        num = str(i)
        for j in range(4 - len(str(i))):
            num = "0" + num
        
        Image.fromarray(data[i]).save(
            target_path / Path(ds_ID + "_" + num + ".png")
        )

# List paths in a txt file
images = os.listdir(target_path)
paths = [target_path / Path(image) for image in images]

with open("train.txt", "w") as f:
    for item in paths:
        f.write("%s\n" % item)
