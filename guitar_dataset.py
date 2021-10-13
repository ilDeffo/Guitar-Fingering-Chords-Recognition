import os
import io
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

label_mappings = {
    'A': 0,
    'B': 1,
    'C': 2,
    'D': 3,
    'E': 4,
    'F': 5,
    'G': 6
}


class GuitarDataset(Dataset):
    """Guitar Dataset"""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(os.listdir(self.root_dir))

    def __getitem__(self, idx):
        """
        Funzione per ottenere un elemento del dataset

        :param idx: Indice dell'elemento
        :return: Una tupla (immagine, label)
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, os.listdir(self.root_dir)[idx])
        image = cv.imread(img_name)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image = np.moveaxis(image, 2, 0)
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image)

        img_base_name = os.path.basename(img_name)
        label = label_mappings.get(img_base_name.split(' ')[0])

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def get_lowest_and_highest_height_and_width_in_dataset(root_dir):
    """
    Funzione per ottenere l'altezza più piccola, l'altezza più grande,
    la larghezza più piccola e la larghezza più grande tra le immagini
    del dataset

    :param root_dir: La cartella contenente tutte le immagini del dataset
    :return: Un dizionario contenente i valori
    """

    d = {
        'lowest_height': 10000,
        'lowest_width': 10000,
        'highest_height': 0,
        'highest_width': 0
    }
    for name in os.listdir(root_dir):
        img_name = os.path.join(root_dir, name)
        x = cv.imread(img_name)
        h, w = x.shape[0], x.shape[1]
        if h < d['lowest_height']:
            d['lowest_height'] = h
        if w < d['lowest_width']:
            d['lowest_width'] = w
        if h > d['highest_height']:
            d['highest_height'] = h
        if w > d['highest_width']:
            d['highest_width'] = w
    return d


def get_mean_and_std_of_dataset(dataset):
    """
    Funzione per ottenere le medie e le std dei valori dei pixel
    delle immagini del dataset per i tre canali RGB

    :param dataset: Il dataset
    :return: Un dizionario contenente i valori
    """
    imgs = [item[0] for item in dataset]
    imgs = torch.stack(imgs, dim=0)

    # calculate mean over each channel (r,g,b)
    mean_r = imgs[:, 0, :, :].mean().item()
    mean_g = imgs[:, 1, :, :].mean().item()
    mean_b = imgs[:, 2, :, :].mean().item()
    mean = [mean_r, mean_g, mean_b]

    # calculate std over each channel (r,g,b)
    std_r = imgs[:, 0, :, :].std().item()
    std_g = imgs[:, 1, :, :].std().item()
    std_b = imgs[:, 2, :, :].std().item()
    std = [std_r, std_g, std_b]

    return {'mean': mean, 'std': std}


if __name__ == '__main__':
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize((540, 959))
    ])
    dataset = GuitarDataset(r'Dataset\all_images', transform=transform)
    print(len(dataset))
    image, label = dataset[100]
    print(image.shape)
    print(label)
    plt.imshow(np.moveaxis(image.numpy(), 0, 2))
    plt.show()

    loader = DataLoader(dataset, 3, shuffle=True)
    dataiter = iter(loader)
    images, labels = next(dataiter)
    print(images.shape)
    print(labels)
    plt.imshow(np.moveaxis(images[0].numpy(), 0, 2))
    plt.show()
