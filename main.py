import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchmetrics import functional as metrics
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, F1Score, AUROC
import numpy as np
import random
from tqdm import tqdm

from dataset import HistopathologicImageDataset
from models.model import Net


device = "cuda" if torch.cuda.is_available() else "cpu"
dataset = HistopathologicImageDataset(labels_file="./data/train_labels.csv", img_dir="./data/train/", transform=transforms.Compose([
    transforms.PILToTensor(),
    transforms.ConvertImageDtype(dtype=torch.float32),
    # TODO: normalize
    #transforms.Normalize([0.5, 0.5, 0.5])#, [0.5, 0.5, 0.5]),
]))


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

#TODO: balance labels?, do image Transformations?
g = torch.Generator()
g.manual_seed(42)
# Use worker_init_fn() and generator to preserve reproducibility
train_set, test_set = random_split(dataset, [176_020, 44_005])
train_loader = DataLoader(dataset=train_set, batch_size=64, shuffle=True, worker_init_fn=seed_worker, generator=g)
test_loader = DataLoader(dataset=test_set, batch_size=64, shuffle=True, worker_init_fn=seed_worker, generator=g)

net = Net(img_dim=[3, 96, 96])
net.to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(net.parameters(), lr=1e-3)


def train():
    for epoch in tqdm(range(10)):
        running_loss = 0.0
        for idx, data in tqdm(enumerate(train_loader, 0)):
            imgs, labels = data[0].to(device), data[1].to(torch.float32).to(device)

            optimizer.zero_grad()

            outputs = net(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if idx % 200 == 199:       # print every 200 mini-batches
                print(f'[{epoch + 1}, {idx + 1:5d}] loss: {running_loss / 200:.3f}')
                running_loss = 0.0

    PATH = "./first_model.pth"
    torch.save(net.state_dict(), PATH)


def evaluate():
    PATH = "./first_model.pth"
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

    metric_collection = MetricCollection([
        Accuracy(),
        Precision(),
        Recall(),
        F1Score(),
        AUROC()
    ])

    print(metric_collection(preds, labels))


if __name__=="__main__": 
    #train()
    evaluate()