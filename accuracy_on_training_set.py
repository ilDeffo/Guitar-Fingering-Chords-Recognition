import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from ChordsClassification.ChordClassificationNetwork_1 import ChordClassificationNetwork
from guitar_dataset import GuitarDataset
from tqdm import tqdm
import copy
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score

saving_path = 'ChordsClassification/saved_models/'

#data_type = "cropped_images"
#data_type = "cropped_processed_images"
#data_type = "cropped_rotated_images"
data_type = "cropped_processed_rotated_images"
#data_type = "cropped_rotated_processed_images_1"
# data_type = "cropped_rotated_processed_images_2"


model = ChordClassificationNetwork()
model.load_state_dict(torch.load(f'{saving_path}{data_type}.pth', map_location=torch.device('cpu')))
model.eval()

transformations = transforms.Compose([
    transforms.Resize((200, 200))
])

training_set = GuitarDataset(f'chords_data/{data_type}/train', transform=transformations)
train_loader = DataLoader(training_set, batch_size=len(training_set))
dataiter = iter(train_loader)
images, labels = next(dataiter)
scores = model(images)
predictions = scores.argmax(1)
preds = predictions.numpy()
y_true = labels.numpy()

print('accuracy on training set: ', accuracy_score(y_true, preds))

# ho salvato i risultati nel file ChordsClassification/results/accuracy_on_training_set.txt

