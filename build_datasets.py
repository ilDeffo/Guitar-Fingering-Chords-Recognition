"""
This script creates 3 different folders of images from our guitar dataset:

- cropped_images: Contains cropped images without any processing.
- cropped_rotated_images: Contains cropped images with the angular correction w.r.t. the strings.
- cropped_processed_rotated_images: Contains cropped images with angular correction w.r.t. the strings and
                                    processing operators applied.

These new datasets are necessary to train, test and evaluate the final classification of chords.
"""

import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision

from guitar_dataset import GuitarDataset
from processing import crop_process_rotate
from Utils.dataset_utils import save_image

# Names of dataset directories
DATASET_DIRS = ["cropped_images", "cropped_rotated_images", "cropped_processed_rotated_images"]

# Base directory containing the dataset directories
BASE_DIR = "chords_data"

if __name__ == '__main__':
    # Directory of our original dataset
    root_dir_guitar_dataset = os.path.join('Dataset', 'all_images')
    guitar_dataset = GuitarDataset(root_dir_guitar_dataset)

    # Creating the base directory for datasets
    if not os.path.exists(BASE_DIR):
        print(f"WARNING! Base directory for custom datasets not found, creating at {BASE_DIR} ...")
        os.mkdir(BASE_DIR)

    # Creating dataset directories
    for d in DATASET_DIRS:
        dir = os.path.join(BASE_DIR, d)
        if not os.path.exists(dir):
            print(f"WARNING! Dataset directory not found, creating at {dir} ...")
            os.mkdir(dir)

    # Iterating over original dataset
    for idx, (image, label) in enumerate(guitar_dataset):
        # Converting image to BGR Pytorch tensor (h, w, c) from the guitar_dataset's image format
        image = np.moveaxis(image.numpy(), 0, 2)
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
        image *= 255
        image = image.round().clip(0, 255)
        image = torch.from_numpy(image)

        # Iterating over datasets directories
        for d in DATASET_DIRS:
            dest_folder = os.path.join(BASE_DIR, d)

            if d == "cropped_images":
                # TODO: Complete code
                continue
            if d == "cropped_rotated_images":
                # TODO: Complete code
                continue
            if d == "cropped_processed_rotated_images":
                cropped_processed_rotated = crop_process_rotate(image)
                save_image(idx, cropped_processed_rotated, label, dest_folder)
                continue
