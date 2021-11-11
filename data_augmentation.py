import os
import io
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

label_mappings = {
    'A': 0,
    'B': 1,
    'C': 2,
    'D': 3,
    'E': 4,
    'F': 5,
    'G': 6
}

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

data_type = "cropped_images"
#data_type = "cropped_processed_images"
#data_type = "cropped_rotated_images"
#data_type = "cropped_processed_rotated_images"
#data_type = "cropped_rotated_processed_images_1"
#data_type = "cropped_rotated_processed_images_2"
#data_type = "cropped_rotated_processed_images_3"
#data_type = "cropped_rotated_processed_images_4"
#data_type = "cropped_rotated_processed_images_5"
#data_type = "cropped_rotated_processed_images_6"

augmented_dataset_dir = f'chords_data/{data_type}_augmented'

if not os.path.exists(augmented_dataset_dir):
    os.mkdir(augmented_dataset_dir)

img_names = [img_name for img_name in os.listdir(f'chords_data/{data_type}') if img_name.endswith('.jpeg')]

transfomations = [
    transforms.ColorJitter(brightness=.5, hue=.3)
]

for idx, img_name in enumerate(img_names):
    img_path = os.path.join(f'chords_data/{data_type}/{img_name}')

    image = cv.imread(img_path)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image = np.moveaxis(image, 2, 0)
    image = image.astype(np.float32) / 255.0
    image = torch.from_numpy(image)

    img_base_name = os.path.basename(img_name)
    label = label_mappings.get(img_base_name.split(' ')[0])

    new_images = [image]
    for t in transfomations:
        new_im = image.clone()
        new_im = t(new_im)
        new_images.append(new_im)

    for idx_1, image in enumerate(new_images):
        image = np.moveaxis(image.numpy(), 0, 2)
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
        image *= 255
        image = image.round().clip(0, 255).astype(np.uint8)
        label_name = get_label_name(label)
        im_name = f'{label_name} ({idx}_{idx_1}).jpeg'
        out_path = f'{augmented_dataset_dir}/{im_name}'
        cv.imwrite(out_path, image)

