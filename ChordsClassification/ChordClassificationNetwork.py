import torch
import torch.nn as nn
import torch.nn.functional as F


class ChordClassificationNetwork(nn.Module):
    def __init__(self, train_model=False):
        super(ChordClassificationNetwork, self).__init__()
        self.train_model = train_model
        self.flatten = nn.Flatten()
        self.firstConv = nn.Conv2d(3, 64, (3, 3))
        self.secondConv = nn.Conv2d(64, 64, (3, 3))
        self.pool = nn.MaxPool2d(2)
        self.drop = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 50 * 50, 256)
        self.fc2 = nn.Linear(256, 256)
        self.outLayer = nn.Linear(256, 7)

    # x represents our data
    def forward(self, x):
        # Pass data through conv1
        x = self.firstConv(x)
        x = F.relu(x)

        x = self.pool(x)

        x = self.secondConv(x)
        x = F.relu(x)

        x = self.pool(x)

        x = self.drop(x)

        # Flatten x with start_dim=1
        x = self.flatten(x, 1)

        # Pass data through fc1
        x = self.fc1(x)
        x = F.relu(x)

        x = self.drop(x)

        x = self.fc2(x)
        x = F.relu(x)

        x = self.drop(x)

        x = self.outLayer(x)

        # Apply softmax to x
        output = F.softmax(x, dim=1)
        return output  # maybe return output.squeeze(1) ?
