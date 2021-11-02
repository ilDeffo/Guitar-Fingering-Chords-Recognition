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
import threading

from guitar_dataset import GuitarDataset
from processing import process_image
from Utils.dataset_utils import save_image

# Names of dataset directories
DATASET_DIRS = ["cropped_images", "cropped_processed_images", "cropped_processed_rotated_images",
                "cropped_rotated_images",
                "cropped_rotated_processed_images_1", "cropped_rotated_processed_images_2"]
# DATASET_DIRS = ["cropped_rotated_processed_images_1", "cropped_rotated_processed_images_2"]
# DATASET_DIRS = ["cropped_rotated_processed_images_2"]

# Base directory containing the dataset directories
BASE_DIR = "chords_data"


def choose_processing_by_dataset_name(dataset_name):
    words = dataset_name.split('_')
    # Default values:
    crop, process, process_mode, rotate, rotate_first = True, False, 0, False, True

    if 'cropped' in words:
        crop = True
    if 'rotated' in words:
        rotate = True
    if 'processed' in words:
        process = True
        if '1' in words:
            process_mode = 1
        if '2' in words:
            process_mode = 2
    if 'processed' in words and 'rotated' in words:
        if words.index('rotated') < words.index('processed'):
            rotate_first = True
    return {'crop': crop, 'rotate': rotate, 'process': process,
            'process_mode': process_mode, 'rotate_first': rotate_first}


def process_and_save(dataset_name, verbose=True):
    dest_folder = os.path.join(BASE_DIR, dataset_name)

    # Choosing processing mode from the dataset_name
    processing_mode = choose_processing_by_dataset_name(dataset_name)
    crop = processing_mode['crop']
    process = processing_mode['process']
    process_mode = processing_mode['process_mode']
    rotate = processing_mode['rotate']
    rotate_first = processing_mode['rotate_first']

    if verbose:
        print(f"*** {dataset_name} ***")
    processed = process_image(image, crop=crop, process=process, rotate=rotate,
                              process_mode=process_mode, rotate_first=rotate_first,
                              verbose=verbose)
    save_image(idx, processed, label, dest_folder)


def save_indexed_processed_image(image, idx, dest_folder,
                                 crop=True, process=True, rotate=True, verbose=True,
                                 process_mode=0, rotate_first=False):

    new_image = process_image(image, crop=crop, process=process, rotate=rotate, verbose=verbose,
                              process_mode=process_mode, rotate_first=rotate_first)

    save_image(idx, new_image, label, dest_folder)


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
    #                                    Set enable_threads to parallelize some processings
    verbose = True
    enable_threads = True
    if enable_threads:
        threads = []
    else:
        threads = None

    # Set enable_restrict_process to true to process only certain difficult images
    enable_restricted_processing = False
    #restricted_processing_indexes = [i for i in range(855, 981)]
    restricted_processing_indexes = [
       863, 897
    ]

    for idx, (image, label) in enumerate(guitar_dataset):
        if enable_restricted_processing:
            if idx not in restricted_processing_indexes:
                continue

        if verbose:
            print(f"------------ Processing and saving of image {idx} ------------")

        # Converting image to BGR Pytorch tensor (h, w, c) from the guitar_dataset's image format
        image = np.moveaxis(image.numpy(), 0, 2)
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
        image *= 255
        image = image.round().clip(0, 255)
        image = torch.from_numpy(image)

        # Reset threads
        if enable_threads:
            threads = []

        # Iterating over datasets directories
        for d in DATASET_DIRS:
            if enable_threads:
                threads.append(
                    threading.Thread(
                    target=process_and_save,
                    kwargs={
                        'dataset_name': d, 'verbose': verbose
                        }, name='process_and_save'))

                threads[-1].start()
            else:
                process_and_save(dataset_name=d, verbose=verbose)

        if enable_threads:
            # Sync threads
            #for t in threads:
            #    t.start()
            for t in threads:
                t.join()

        if verbose:
            print(f"-------------------------------------------------------------\n")

