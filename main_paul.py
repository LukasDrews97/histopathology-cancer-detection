import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, F1Score, AUROC
import numpy as np
import random
from tqdm import tqdm

from dataset import HistopathologicImageDataset
# Mi modelo
from models.cnn_paul import Net

PATH = f"./cnn_paul.pth"


device = "cuda" if torch.cuda.is_available() else "cpu"
dataset = HistopathologicImageDataset(
    labels_file="./data/train_labels.csv",
    img_dir="./data/train/",
    transform=transforms.Compose(
        [
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(dtype=torch.float32),
        ]
    ),
)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# TODO: balance labels?, do image Transformations?
g = torch.Generator()
g.manual_seed(42)
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
# Use worker_init_fn() and generator to preserve reproducibility
train_set, test_set = random_split(dataset, [176_020, 44_005])
train_loader = DataLoader(
    dataset=train_set,
    batch_size=128,
    shuffle=True,
    worker_init_fn=seed_worker,
    generator=g,
)
test_loader = DataLoader(
    dataset=test_set,
    batch_size=128,
    shuffle=True,
    worker_init_fn=seed_worker,
    generator=g,
)

net = Net(img_dim=[3, 96, 96])
net.to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(net.parameters(), lr=1e-3)

def train():
    #net.load_state_dict(torch.load(PATH))

    for epoch in tqdm(range(10)):
        current_loss = torch.float32

        for i, data in enumerate(train_loader, 0):
            images, labels = data[0], data[1].to(torch.float32)
            images = images.to(device)
            labels = labels.to(device)

            outputs = net(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            current_loss += loss.item()
            if i % 100 == 0:  # print every 100 mini-batches
                print(f"Epoch [{epoch+1}, {10}], Step:[{i+1}], Loss: {current_loss/200:.4f}")
                current_loss = torch.float32

    torch.save(net.state_dict(), PATH)

def evaluate():
    #net.load_state_dict(torch.load(PATH))
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
    metric_collection = MetricCollection(
        [Accuracy(), Precision(), Recall(), F1Score(), AUROC()]
    )
    print(metric_collection(preds, labels))

if __name__=="__main__": 
    train()
    evaluate()