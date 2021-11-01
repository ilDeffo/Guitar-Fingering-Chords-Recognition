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
    threads = []

    # Set enable_restrict_process to true to process only certain images
    enable_restricted_processing = True
    # restricted_processing_indexes = [i for i in range(503, 542)]
    restricted_processing_indexes = [120]

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
            dest_folder = os.path.join(BASE_DIR, d)

            if d == "cropped_images":
                if verbose:
                    print("*** cropping ***")

                '''
                threads_executor.submit(
                    save_indexed_processed_image,
                    kwargs={
                        'image': image, 'idx': idx,
                        'dest_folder': dest_folder,
                        'crop': True, 'process': False,
                        'rotate': False, 'verbose': verbose
                    })
                '''
                if enable_threads:
                    threads.append(
                        threading.Thread(
                            target=save_indexed_processed_image,
                            kwargs={
                                'image': image, 'idx': idx,
                                'dest_folder': dest_folder,
                                'crop': True, 'process': False,
                                'rotate': False, 'verbose': verbose
                            }, name='save_indexed_processed_image'))
                else:
                    cropped_processed_rotated = process_image(image, crop=True, process=False, rotate=False, verbose=verbose)
                    save_image(idx, cropped_processed_rotated, label, dest_folder)
                continue
            if d == "cropped_processed_images":
                if verbose:
                    print("*** cropping and processing ***")
                '''
                threads_executor.submit(
                    save_indexed_processed_image,
                    kwargs={
                        'image': image, 'idx': idx,
                        'dest_folder': dest_folder,
                        'crop': True, 'process': True,
                        'rotate': False, 'verbose': verbose
                    })
                '''
                if enable_threads:
                    threads.append(
                        threading.Thread(
                            target=save_indexed_processed_image,
                            kwargs={
                                'image': image, 'idx': idx,
                                'dest_folder': dest_folder,
                                'crop': True, 'process': True,
                                'rotate': False, 'verbose': verbose
                            }, name='save_indexed_processed_image'))
                else:
                    cropped_processed = process_image(image, crop=True, process=True, rotate=False, verbose=verbose)
                    save_image(idx, cropped_processed, label, dest_folder)
                continue
            if d == "cropped_rotated_images":
                if verbose:
                    print("*** cropping and rotating ***")
                '''
                threads_executor.submit(
                    save_indexed_processed_image,
                    kwargs={
                        'image': image, 'idx': idx,
                        'dest_folder': dest_folder,
                        'crop': True, 'process': False,
                        'rotate': True, 'verbose': verbose
                    })
                '''
                if enable_threads:
                    threads.append(
                        threading.Thread(
                            target=save_indexed_processed_image,
                            kwargs={
                                'image': image, 'idx': idx,
                                'dest_folder': dest_folder,
                                'crop': True, 'process': False,
                                'rotate': True, 'verbose': verbose
                            }, name='save_indexed_processed_image'))
                else:
                    cropped_processed_rotated = process_image(image, crop=True, process=False, rotate=True, verbose=verbose)
                    save_image(idx, cropped_processed_rotated, label, dest_folder)
                continue
            if d == "cropped_processed_rotated_images":
                if verbose:
                    print("*** cropping, processing and rotating ***")
                '''
                threads_executor.submit(
                    save_indexed_processed_image,
                    kwargs={
                        'image': image, 'idx': idx,
                        'dest_folder': dest_folder,
                        'crop': True, 'process': True,
                        'rotate': True, 'verbose': verbose
                    })
                '''
                if enable_threads:
                    threads.append(
                        threading.Thread(
                            target=save_indexed_processed_image,
                            kwargs={
                                'image': image, 'idx': idx,
                                'dest_folder': dest_folder,
                                'crop': True, 'process': True,
                                'rotate': True, 'verbose': verbose
                            }, name='save_indexed_processed_image'))
                else:
                    cropped_processed_rotated = process_image(image, crop=True, process=True, rotate=True, verbose=verbose)
                    save_image(idx, cropped_processed_rotated, label, dest_folder)
                continue
            if d == "cropped_rotated_processed_images_1":
                if verbose:
                    print("*** cropping, rotating and processing (mode 1) ***")
                '''
                threads_executor.submit(
                    save_indexed_processed_image,
                    kwargs={
                        'image': image, 'idx': idx,
                        'dest_folder': dest_folder,
                        'crop': True, 'process': True,
                        'rotate': True, 'process_mode': 1,
                        'rotate_first': True, 'verbose': verbose
                    })
                '''
                if enable_threads:
                    threads.append(
                        threading.Thread(
                            target=save_indexed_processed_image,
                            kwargs={
                                'image': image, 'idx': idx,
                                'dest_folder': dest_folder,
                                'crop': True, 'process': True,
                                'rotate': True, 'process_mode': 1,
                                'rotate_first': True, 'verbose': verbose
                            }, name='save_indexed_processed_image'))
                else:
                    cropped_rotated_processed_1 = process_image(image, crop=True, process=True, process_mode=1, rotate=True, rotate_first=True, verbose=verbose)
                    save_image(idx, cropped_rotated_processed_1, label, dest_folder)
                continue
            if d == "cropped_rotated_processed_images_2":
                if verbose:
                    print("*** cropping, rotating and processing (mode 2) ***")
                '''
                threads_executor.submit(
                    save_indexed_processed_image,
                    kwargs={
                        'image': image, 'idx': idx,
                        'dest_folder': dest_folder,
                        'crop': True, 'process': True,
                        'rotate': True, 'process_mode': 2,
                        'rotate_first': True, 'verbose': verbose
                    })
                '''
                if enable_threads:
                    threads.append(
                        threading.Thread(
                            target=save_indexed_processed_image,
                            kwargs={
                                'image': image, 'idx': idx,
                                'dest_folder': dest_folder,
                                'crop': True, 'process': True,
                                'rotate': True, 'process_mode': 2,
                                'rotate_first': True, 'verbose': verbose
                            }, name='save_indexed_processed_image'))
                else:
                    cropped_rotated_processed_2 = process_image(image, crop=True, process=True, process_mode=2, rotate=True, rotate_first=True, verbose=verbose)
                    save_image(idx, cropped_rotated_processed_2, label, dest_folder)
                continue

        if enable_threads:
            # Start and sync threads
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        if verbose:
            print(f"-------------------------------------------------------------\n")

