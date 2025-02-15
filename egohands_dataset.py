import os
import io
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class EgoHandsDataset(Dataset):
    '''EgoHands dataset'''
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.images_dir = os.path.join(root_dir, 'all_images')
        self.bounding_boxes_file_name = os.path.join(root_dir, 'all_images_bounding_boxes.csv')
        self.bounding_boxes = pd.read_csv(self.bounding_boxes_file_name)
        self.transform = transform

    def __len__(self):
        return len(os.listdir(self.images_dir))

    def __getitem__(self, idx):
        '''
        Funzione per ottenere un elemento del dataset

        :param idx: Indice dell'elemento
        :return: Una tupla (immagine, bounding_boxes) dove bounding_boxes è un tensore
        '''
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x = self.bounding_boxes.iloc[idx]
        img_name = os.path.join(self.images_dir, x['image_name'])
        image = cv.imread(img_name)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image = np.moveaxis(image, 2, 0)
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image)

        if self.transform:
            image = self.transform(image)

        bounding_boxes = torch.zeros((4, 4))
        bounding_boxes[0] = torch.from_numpy(
            x.loc[['box0_x1', 'box0_y1', 'box0_x2', 'box0_y2']].values.astype(np.float32))
        bounding_boxes[1] = torch.from_numpy(
            x.loc[['box1_x1', 'box1_y1', 'box1_x2', 'box1_y2']].values.astype(np.float32))
        bounding_boxes[2] = torch.from_numpy(
            x.loc[['box2_x1', 'box2_y1', 'box2_x2', 'box2_y2']].values.astype(np.float32))
        bounding_boxes[3] = torch.from_numpy(
            x.loc[['box3_x1', 'box3_y1', 'box3_x2', 'box3_y2']].values.astype(np.float32))

        bounding_boxes = bounding_boxes[~torch.isnan(bounding_boxes[:, 0]), :]

        target = {
            'boxes': bounding_boxes,
            'labels': torch.full((bounding_boxes.shape[0],), 1)
        }

        return image, target

    @staticmethod
    def egohands_collate_fn(data):
        tmp = tuple(zip(*data))
        images = tmp[0]
        images = torch.stack(images, dim=0)
        return images, tmp[1]

if __name__ == '__main__':
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((720, 1280))
    ])
    dataset = EgoHandsDataset('egohands_data')
    print(len(dataset))
    image, target = dataset[0]
    print(image.shape)
    print(target)
    #plt.imshow(np.moveaxis(image.numpy(), 0, 2))
    #plt.show()

    loader = DataLoader(dataset, 3, shuffle=True, collate_fn=EgoHandsDataset.egohands_collate_fn)
    dataiter = iter(loader)
    images, targets = next(dataiter)
    print(images.shape)
    print(targets)
    #plt.imshow(np.moveaxis(images[0].numpy(), 0, 2))
    #plt.show()

    image, target = dataset[0]
    im = np.moveaxis(image.numpy(), 0, 2)
    for box in target['boxes']:
        box = box.numpy().astype(int)
        cv.rectangle(im, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
    plt.imshow(cv.cvtColor(im, cv.COLOR_BGR2RGB))
    plt.show()