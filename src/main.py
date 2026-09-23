import math
import numpy as np
import matplotlib.pyplot as plt

from modules.read_data import read_data

### CONSTANTS
IMG_DIR_PATH = "./imgs"

### FUNCTIONS

def save_points(data_points):
    """
    Saves the data point(s) in a square grayscale image file.
    """

    for idx, data_point in enumerate(data_points):
        img_w = int(math.sqrt(data_point.shape[0]))
        img = np.reshape(data_point, shape=(img_w, img_w))

        plt.imshow(img, cmap="gray", vmin=0, vmax=1)
        plt.axis("off")
        plt.savefig(f"{IMG_DIR_PATH}/data_point_{idx}.png", bbox_inches="tight", pad_inches=0.1)
        # plt.close()

# Read in pickled dataset
train_data, val_data, test_data = read_data()
print("--- LOADED DATA Points ---")
print(f"{train_data[0].shape[0]} Training Points | {val_data[0].shape[0]} Validation Points | {test_data[0].shape[0]} Test Points")

print("\n")

# # Display some data points
# save_points(train_data[0][5:10])


