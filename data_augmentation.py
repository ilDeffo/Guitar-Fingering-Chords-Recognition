import os
import io
import random
import re

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


#data_type = "cropped_images"
# data_type = "cropped_processed_images"
#data_type = "cropped_rotated_images"
# data_type = "cropped_processed_rotated_images"
# data_type = "cropped_rotated_processed_images_1"
# data_type = "cropped_rotated_processed_images_2"
# data_type = "cropped_rotated_processed_images_3"
# data_type = "cropped_rotated_processed_images_4"
data_type = "cropped_rotated_processed_images_5"
# data_type = "cropped_rotated_processed_images_6"

extended_dataset_dir = f'chords_data/{data_type}_extended/train'

if __name__ == "__main__":
    if not os.path.exists(extended_dataset_dir):
        os.mkdir(extended_dataset_dir)

    img_names = [img_name for img_name in os.listdir(f'chords_data/{data_type}_extended/train_not_augmented') if
                 img_name.endswith('.jpeg')
                 and not os.path.isdir(img_name)]

    # transfomations = [
    #     transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.25),
    #     transforms.GaussianBlur(5),
    #     transforms.RandomAdjustSharpness(3, 1)
    # ]

    for idx, img_name in enumerate(img_names):
        img_path = f'chords_data/{data_type}_extended/train_not_augmented/{img_name}'

        image = cv.imread(img_path)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image = np.moveaxis(image, 2, 0)
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image)

        img_base_name = os.path.basename(img_name)
        index_name = re.search('\((.*?)\)', img_base_name).group(1)
        label = label_mappings.get(img_base_name.split(' ')[0])
        label_name = get_label_name(label)

        for idx_1, t in enumerate([
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.25),
            #transforms.RandomAutocontrast(),
            transforms.GaussianBlur(random.randrange(3, 16, 2)),
            transforms.RandomAdjustSharpness(random.uniform(1.5, 3), 1)]
        ):
            new_im = image.clone()
            new_im = t(new_im)
            new_im = np.moveaxis(new_im.numpy(), 0, 2)
            new_im = cv.cvtColor(new_im, cv.COLOR_RGB2BGR)
            new_im *= 255
            new_im = new_im.round().clip(0, 255).astype(np.uint8)
            im_name = f'{label_name} ({index_name}_{idx_1 + 1}).jpeg'
            out_path = f'{extended_dataset_dir}/{im_name}'
            cv.imwrite(out_path, new_im)

        image = np.moveaxis(image.numpy(), 0, 2)
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
        image *= 255
        image = image.round().clip(0, 255).astype(np.uint8)
        im_name = f'{label_name} ({index_name}_0).jpeg'
        out_path = f'{extended_dataset_dir}/{im_name}'
        cv.imwrite(out_path, image)
