import torch
import torch.nn as nn
import torch.nn.functional as F


class ChordClassificationNetwork(nn.Module):
    def __init__(self):
        super(ChordClassificationNetwork, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, (3, 3)),
            nn.ReLU(),
            nn.MaxPool2d(3),
            nn.Conv2d(64, 128, (3, 3)),
            nn.ReLU(),
            nn.MaxPool2d(3)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(56448, 7),
        )

    # x represents our data
    def forward(self, x):
        x = self.features(x)
        #print(x.shape)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        output = F.softmax(x, dim=1)
        return output.squeeze(1)  # maybe return output.squeeze(1) ?

