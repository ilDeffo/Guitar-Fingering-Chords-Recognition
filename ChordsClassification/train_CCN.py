import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from ChordClassificationNetwork import ChordClassificationNetwork
from guitar_dataset import GuitarDataset
from tqdm import tqdm

data_type = "cropped_images"
# data_type = "cropped_rotated_images"
# data_type = "cropped_processed_rotated_images"

transformations = transforms.Compose([
    transforms.Resize((300, 300))
])

num_epochs = 10
learning_rate = 0.001
train_CNN = False
batch_size = 32
shuffle = True
pin_memory = True
num_workers = 1

dataset = GuitarDataset(f"../chords_data/{data_type}/train", transform=transformations)
train_set, validation_set = torch.utils.data.random_split(dataset, [int(0.8 * len(dataset)),
                                                                    len(dataset) - int(0.8 * len(dataset))])
train_loader = DataLoader(dataset=train_set, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                          pin_memory=pin_memory)
validation_loader = DataLoader(dataset=validation_set, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                               pin_memory=pin_memory)

testset = GuitarDataset(f"../chords_data/{data_type}/test", transform=transformations)
test_loader = DataLoader(dataset=testset, shuffle=False, batch_size=batch_size, num_workers=num_workers,
                         pin_memory=pin_memory)

PATH = f"./saved_models/{data_type}.pth"



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
    for epoch in range(num_epochs + 1):
        loop = tqdm(train_loader, total=len(train_loader), leave=True)
        # if epoch % 2 == 0:
        loop.set_postfix(val_acc=check_accuracy(validation_loader, model))
        if epoch == num_epochs:
            break
        for imgs, labels in loop:
            labels = torch.nn.functional.one_hot(labels, num_classes=7).float()
            imgs = imgs.to(device)
            labels = labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loop.set_description(f"Epoch [{epoch + 1}/{num_epochs}]")
            loop.set_postfix(loss=loss.item())

    torch.save(model.state_dict(), PATH)


def test():
    model.load_state_dict(torch.load(PATH))

    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the test images: %d %%' % (
            100 * correct / total))


if __name__ == "__main__":
    print(f"Working on {data_type}")

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.empty_cache()
        print(f"GPU found! Using {device}...")

    model = ChordClassificationNetwork().to(device)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train()
    test()
