import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, F1Score, AUROC
import numpy as np
import pandas as pd
import random
from tqdm import tqdm

from dataset import HistopathologicImageDataset
from models.cnn_2 import Net


device = "cuda" if torch.cuda.is_available() else "cpu"
dataset = HistopathologicImageDataset(labels_file="./data/train_labels.csv", img_dir="./data/train/", transform=transforms.Compose([
    transforms.PILToTensor(),
    transforms.ConvertImageDtype(dtype=torch.float32),
    transforms.Normalize([0.7025, 0.5463, 0.6965], [0.2389, 0.2821, 0.2163]),
]))


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

#TODO: balance labels?, do image Transformations?
g = torch.Generator()
g.manual_seed(42)
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
# Use worker_init_fn() and generator to preserve reproducibility
train_set, test_set = random_split(dataset, [176_020, 44_005])
train_loader = DataLoader(dataset=train_set, batch_size=64, shuffle=True, worker_init_fn=seed_worker, generator=g)
test_loader = DataLoader(dataset=test_set, batch_size=64, shuffle=False, worker_init_fn=seed_worker, generator=g)


net = Net(img_dim=[3, 96, 96])
net.to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(net.parameters(), lr=1e-3)

metric_collection = MetricCollection([
    Accuracy(),
    Precision(),
    Recall(),
    F1Score(),
    AUROC()
])


def train(name):
    losses = []
    accuracy = []
    precision = []
    recall = []
    f1_score = []
    auroc = []

    for epoch in tqdm(range(10)):
        running_loss = 0.0
        list_of_preds = []
        list_of_labels = []

        for idx, data in tqdm(enumerate(train_loader, 0)):
            imgs, labels = data[0].to(device), data[1].to(torch.float32).to(device)

            optimizer.zero_grad()

            preds = net(imgs)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            list_of_preds.append(preds)
            list_of_labels.append(labels)

            if idx % 200 == 199: # calculate and print train metrics every 200 mini-batches
                loss = running_loss / 200
                losses.append(loss)

                metrics = metric_collection(torch.cat(list_of_preds).cpu(), torch.cat(list_of_labels).cpu().to(torch.int8))
                accuracy.append(metrics["Accuracy"].item())
                precision.append(metrics["Precision"].item())
                recall.append(metrics["Recall"].item())
                f1_score.append(metrics["F1Score"].item())
                auroc.append(metrics["AUROC"].item())

                print(f'[{epoch + 1}, {idx + 1:5d}] loss: {loss:.3f}')
                print(f'[{epoch + 1}, {idx + 1:5d}] accuracy: {accuracy[-1]:.3f} precision: {precision[-1]:.3f} recall: {recall[-1]:.3f} f1_score: {f1_score[-1]:.3f} auroc: {auroc[-1]:.3f}')
                running_loss = 0.0
                list_of_labels.clear()
                list_of_preds.clear()

    train_metrics = pd.DataFrame(
        {'bce_loss': losses, 
        'accuracy': accuracy, 
        'precision': precision, 
        'recall': recall,
        'f1_score': f1_score,
        'auroc': auroc
        })
    train_metrics.to_csv(f'{name}_train.csv')
    PATH = f"./{name}.pth"
    torch.save(net.state_dict(), PATH)


def evaluate(name):
    PATH = f"./{name}.pth"
    net.load_state_dict(torch.load(PATH))
    net.cpu()

    list_of_labels = []
    list_of_preds = []


    with torch.no_grad():
        for _, data in tqdm(enumerate(test_loader)):
            imgs, labels = data
            preds = net(imgs)
            list_of_preds.append(preds)
            list_of_labels.append(labels)


    labels = torch.cat(list_of_labels)
    preds = torch.cat(list_of_preds)
    metrics = metric_collection(preds, labels)
    print(metrics)
    test_metrics = pd.DataFrame({
        'accuracy': metrics['Accuracy'].item(),
        'precision': metrics['Precision'].item(),
        'recall': metrics['Recall'].item(),
        'f1_score': metrics['F1Score'].item(),
        'auroc': metrics['AUROC'].item()
        }, index=[0])
    test_metrics.to_csv(f'{name}_test.csv')



if __name__=="__main__": 
    #mean, std = calculate_mean_std()
    #print(mean)
    #print(std)
    name = "cnn2"
    train(name)
    evaluate(name)