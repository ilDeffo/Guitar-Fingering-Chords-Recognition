import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision

from guitar_dataset import GuitarDataset


def get_label_name(label):
    """
    Method to get name of class, given the numeric code.

    :param label: Numeric code of the class.
    :return: The label name associated to the class.
    """
    from guitar_dataset import label_mappings
    for k, v in label_mappings.items():
        if v == label:
            return k


def save_image(idx, image, label, dest_folder):
    """
    Method to save an image in directory.

    :param idx: Index of the image in the dataset.
    :param image: BGR PyTorch image tensor of shape (h, w, c).
    :param label: Numeric code for the class of the image.
    :param dest_folder: Destination folder to store image.
    :return: Nothing.
    """

    label_name = get_label_name(label)
    im_name = label_name + ' (' + str(idx) + ')' + '.jpeg'
    out_path = os.path.join(dest_folder, im_name)
    cv.imwrite(out_path, image.numpy())
