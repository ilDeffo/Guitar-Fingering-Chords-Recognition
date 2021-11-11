import os
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from ChordsClassification.ChordClassificationNetwork_1 import ChordClassificationNetwork
from guitar_dataset import GuitarDataset
from tqdm import tqdm
import copy

# data_type = "cropped_images"
#data_type = "cropped_processed_images"
data_type = "cropped_rotated_images"
#data_type = "cropped_processed_rotated_images"
#data_type = "cropped_rotated_processed_images_1"
#data_type = "cropped_rotated_processed_images_2"
#data_type = "cropped_rotated_processed_images_3"
#data_type = "cropped_rotated_processed_images_4"
#data_type = "cropped_rotated_processed_images_5"
#data_type = "cropped_rotated_processed_images_6"

transformations = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.RandomApply([
        transforms.ColorJitter(brightness=0.2, contrast=0, saturation=0, hue=0)], 0.25),
    transforms.RandomApply([
        transforms.ColorJitter(brightness=0, contrast=0.2, saturation=0, hue=0)], 0.25),
    transforms.RandomApply([
        transforms.ColorJitter(brightness=0, contrast=0, saturation=0.2, hue=0)], 0.25),
    transforms.RandomApply([
        transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0.2)], 0.25),
    transforms.RandomApply([
        transforms.GaussianBlur(3)], 0.25),
    transforms.RandomAdjustSharpness(random.uniform(0.5, 1.5), 0.25)
])

num_epochs = 20
learning_rate = 0.001
train_CNN = False
batch_size = 32
shuffle = True
pin_memory = True
num_workers = 4

trainset = GuitarDataset(f"../chords_data/{data_type}/train", transform=transformations)

train_loader = DataLoader(dataset=trainset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                          pin_memory=pin_memory)

testset = GuitarDataset(f"../chords_data/{data_type}/test", transform=transformations)
test_loader = DataLoader(dataset=testset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                         pin_memory=pin_memory)

PATH = f"./saved_models/{data_type}.pth"

results_dir = 'results/'
if not os.path.exists(results_dir):
    os.mkdir(results_dir)

if __name__ == "__main__":
    print(f"Working on {data_type}")

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"GPU found! Using {device}...")
        torch.cuda.empty_cache()
        # os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64'

    model = ChordClassificationNetwork().to(device)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    import csv

    csv_header = ['Classification loss']

    csvfile = open(results_dir + 'augmented_' + data_type + '_our_nn_loss.csv', 'w', newline='')
    writer = csv.writer(csvfile)
    writer.writerow(csv_header)

    print('------------- Training -----------------')
    best_acc = 0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        print('-' * 10)
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                for i, (imgs, labels) in enumerate(tqdm(train_loader)):
                    labels = torch.nn.functional.one_hot(labels, num_classes=7).float()
                    imgs = imgs.to(device)
                    labels = labels.to(device)
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)
                    # writing loss in csv file
                    row = [loss.item()]
                    writer.writerow(row)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            if phase == 'val':
                model.eval()  # Set model to evaluate mode
                num_correct = 0
                num_samples = 0

                with torch.no_grad():
                    for x, y in test_loader:
                        x = x.to(device=device)
                        y = y.to(device=device)

                        scores = model(x)
                        # predictions = torch.tensor([1.0 if i >= 0.5 else 0.0 for i in scores]).to(device)
                        predictions = scores.argmax(1)
                        num_correct += (predictions == y).sum()
                        num_samples += predictions.size(0)
                        acc = float(num_correct) / float(num_samples) * 100
                        print(
                            f"Got {num_correct} / {num_samples} with accuracy {acc:.2f}"
                        )
                    if acc > best_acc:
                        best_acc = acc
                        best_model_wts = copy.deepcopy(model.state_dict())

    csvfile.close()
    torch.save(best_model_wts, PATH)

    print('--------- Evaluating performance and saving results ------------')
    model = ChordClassificationNetwork()
    model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))

    testloader = DataLoader(dataset=testset, shuffle=False, batch_size=batch_size, num_workers=num_workers)
    preds = []
    y_true = []
    for x, y in test_loader:
        scores = model(x)
        predictions = scores.argmax(1)
        preds.append(predictions)
        y_true.append(y)

    preds = torch.cat(preds)
    y_true = torch.cat(y_true)

    import pandas as pd

    df = pd.DataFrame({'predictions': preds.cpu().numpy(), 'y_true': y_true.cpu().numpy()})
    df.to_csv(results_dir + data_type + '_our_nn_predictions__ytrue.csv', index=False)
    preds = df['predictions'].values
    y_true = df['y_true'].values
    reverse_label_mappings = {
        0: 'A',
        1: 'B',
        2: 'C',
        3: 'D',
        4: 'E',
        5: 'F',
        6: 'G'
    }
    results_dict = {}
    for i in range(7):
        results_dict[reverse_label_mappings[i]] = []
        results_dict[reverse_label_mappings[i]].append(((preds == y_true) & (y_true == i)).sum())
        results_dict[reverse_label_mappings[i]].append((y_true == i).sum())
    df = pd.DataFrame(results_dict, index=['num_correct', 'num_samples'])
    df = df.T
    df.to_csv(results_dir + 'augmented_' + data_type + '_our_nn_performances.csv')

    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import f1_score

    print(confusion_matrix(y_true, preds))
    accuracy = accuracy_score(y_true, preds)
    print('Accuracy:', accuracy)
    with open(results_dir + 'augmented_' + data_type + '_our_nn_accuracy.txt', 'wt') as f:
        f.write(str(accuracy))

    precision = precision_score(y_true, preds, average=None)
    print('Precision:', precision)

    recall = recall_score(y_true, preds, average=None)
    print('Recall:', recall)

    f1_score = f1_score(y_true, preds, average=None)
    print('F1 score:', f1_score)

    df = pd.DataFrame({
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }, index=['A', 'B', 'C', 'D', 'E', 'F', 'G'])
    df.to_csv(results_dir + 'augmented_' + data_type + '_our_nn_precision_recall_f1_score.csv')
