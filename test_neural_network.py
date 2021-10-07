import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from torch.utils.data import DataLoader

from egohands_dataset import EgoHandsDataset
import torchvision
from torchvision import models
from torchvision import transforms
from torchvision import datasets

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

dataset = EgoHandsDataset('egohands_data')

image, target = dataset[10]
image = image.unsqueeze(0)
target = (target,)
#x = model(image, target)
#print(x)

loader = DataLoader(dataset, batch_size=3, shuffle=True, collate_fn=EgoHandsDataset.egohands_collate_fn)
dataiter = iter(loader)
images, targets = next(dataiter)
out = model(images, targets)
print(out)
