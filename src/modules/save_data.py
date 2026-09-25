import math
import numpy as np
import matplotlib.pyplot as plt
from src.modules.constants import IMG_DIR_PATH

def save_points(data_points, name="data_point"):
    """
    Saves the data point(s) in a square grayscale image file.
    """

    for idx, data_point in enumerate(data_points):
        img_w = int(math.sqrt(data_point.shape[0]))
        img = np.reshape(data_point, shape=(img_w, img_w))

        plt.imshow(img, cmap="gray", vmin=0, vmax=1)
        plt.axis("off")
        fig_name = f"{IMG_DIR_PATH}/{name}"
        if len(data_points) > 1:
            fig_name += f"-{idx}"
        plt.savefig(f"{fig_name}.png", bbox_inches="tight", pad_inches=0.1)