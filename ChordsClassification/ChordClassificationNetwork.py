import torch
import torch.nn as nn
import torch.nn.functional as F


class ChordClassificationNetwork(nn.Module):
    def __init__(self):
        super(ChordClassificationNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.firstConv = nn.Conv2d(3, 64, (3, 3))
        self.secondConv = nn.Conv2d(64, 64, (3, 3))
        self.pool = nn.MaxPool2d(2)
        self.drop = nn.Dropout(0.50)
        self.fc1 = nn.Linear(147456, 512)
        self.fc2 = nn.Linear(512, 64)
        self.outLayer = nn.Linear(64, 7)

    # x represents our data
    def forward(self, x):
        #print(x.shape)
        # Pass data through conv1
        x = self.firstConv(x)
        #print(x.shape)
        x = F.relu(x)
        #print(x.shape)

        x = self.pool(x)
        #print(x.shape)

        x = self.secondConv(x)
        #print(x.shape)
        x = F.relu(x)
        #print(x.shape)

        x = self.pool(x)
        #print(x.shape)

        x = self.drop(x)
        #print(x.shape)

        # Flatten x with start_dim=1
        x = self.flatten(x)
        #print(x.shape)

        # Pass data through fc1
        x = self.fc1(x)
        #print(x.shape)
        x = F.relu(x)
        #print(x.shape)

        x = self.drop(x)
        #print(x.shape)

        x = self.fc2(x)
        #print(x.shape)
        x = F.relu(x)
        #print(x.shape)

        x = self.drop(x)
        #print(x.shape)

        x = self.outLayer(x)
        #print(x.shape)

        # Apply softmax to x
        output = F.softmax(x, dim=1)
        return output.squeeze(1)  # maybe return output.squeeze(1) ?

