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
from processing import process_image
from Utils.dataset_utils import save_image

# Names of dataset directories
DATASET_DIRS = ["cropped_images", "cropped_rotated_images", "cropped_processed_rotated_images",
                "cropped_rotated_processed_images_1", "cropped_rotated_processed_images_2"]
# DATASET_DIRS = ["cropped_rotated_processed_images_1", "cropped_rotated_processed_images_2"]

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

    # Iterating over original dataset -> Set verbose to true to see all the outputs
    verbose = True
    for idx, (image, label) in enumerate(guitar_dataset):
        if verbose:
            print(f"------------ Processing and saving of image {idx} ------------")

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
                if verbose:
                    print("*** cropping ***")
                cropped_processed_rotated = process_image(image, crop=True, process=False, rotate=False, verbose=verbose)
                save_image(idx, cropped_processed_rotated, label, dest_folder)
                continue
            if d == "cropped_rotated_images":
                if verbose:
                    print("*** cropping and rotating ***")
                cropped_processed_rotated = process_image(image, crop=True, process=False, rotate=True, verbose=verbose)
                save_image(idx, cropped_processed_rotated, label, dest_folder)
                continue
            if d == "cropped_processed_rotated_images":
                if verbose:
                    print("*** cropping, processing and rotating ***")
                cropped_processed_rotated = process_image(image, crop=True, process=True, rotate=True, verbose=verbose)
                save_image(idx, cropped_processed_rotated, label, dest_folder)
                continue
            if d == "cropped_rotated_processed_images_1":
                if verbose:
                    print("*** cropping, rotating and processing (mode 1) ***")
                cropped_rotated_processed_1 = process_image(image, crop=True, process=True, process_mode=1, rotate=True, verbose=verbose)
                save_image(idx, cropped_rotated_processed_1, label, dest_folder)
                continue

            if d == "cropped_rotated_processed_images_2":
                if verbose:
                    print("*** cropping, rotating and processing (mode 2) ***")
                cropped_rotated_processed_2 = process_image(image, crop=True, process=True, process_mode=2, rotate=True, verbose=verbose)
                save_image(idx, cropped_rotated_processed_2, label, dest_folder)
                continue

        if verbose:
            print(f"------------------------------------------------------------\n")
