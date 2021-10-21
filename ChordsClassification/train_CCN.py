import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from ChordClassificationNetwork import ChordClassificationNetwork
from guitar_dataset import GuitarDataset
from tqdm import tqdm

device = ("cuda" if torch.cuda.is_available() else "cpu")

transformations = transforms.Compose([
     transforms.Resize((100, 100))
])

num_epochs = 10
learning_rate = 0.001
train_CNN = False
batch_size = 32
shuffle = True
pin_memory = True
num_workers = 1

dataset = GuitarDataset("../chords_data/cropped_images/train", transform=transformations)
train_set, validation_set = torch.utils.data.random_split(dataset, [int(0.8 * len(dataset)), len(dataset) - int(0.8*len(dataset))])
train_loader = DataLoader(dataset=train_set, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                          pin_memory=pin_memory)
validation_loader = DataLoader(dataset=validation_set, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                               pin_memory=pin_memory)

model = ChordClassificationNetwork().to(device)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


def check_accuracy(loader, model):
    if loader == train_loader:
        print("Checking accuracy on training data")
    else:
        print("Checking accuracy on validation data")

    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            # predictions = torch.tensor([1.0 if i >= 0.5 else 0.0 for i in scores]).to(device)
            predictions = scores.argmax(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
            print(
                f"Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}"
            )
    return f"{float(num_correct) / float(num_samples) * 100:.2f}"


def train():
    model.train()
    for epoch in range(num_epochs):
        loop = tqdm(train_loader, total=len(train_loader), leave=True)
        if epoch % 2 == 0:
            loop.set_postfix(val_acc=check_accuracy(validation_loader, model))
        for imgs, labels in loop:
            labels = torch.nn.functional.one_hot(labels).float()
            imgs = imgs.to(device)
            labels = labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loop.set_description(f"Epoch [{epoch}/{num_epochs}]")
            loop.set_postfix(loss=loss.item())


if __name__ == "__main__":
    train()
