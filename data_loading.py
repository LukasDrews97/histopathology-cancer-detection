# Preprocessing, Data Iterators

import random
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from dataset import HistopathologicImageDataset

def get_train_and_test_loader(labels_file=None, img_dir=None):
    '''load dataset and return dataloader for training and testing '''
    if labels_file is None:
        labels_file="./data/train_labels.csv"
    if img_dir is None:
        img_dir="./data/train/"
    dataset = HistopathologicImageDataset(labels_file=labels_file, img_dir=img_dir, transform=transforms.Compose([
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(dtype=torch.float32),
        transforms.Normalize([0.7025, 0.5463, 0.6965], [0.2389, 0.2821, 0.2163]),
    ]))

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(42)
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    # Use worker_init_fn() and generator to preserve reproducibility
    train_set, test_set = random_split(dataset, [176_020, 44_005])
    train_loader = DataLoader(dataset=train_set, batch_size=64, shuffle=True, worker_init_fn=seed_worker, generator=g)
    test_loader = DataLoader(dataset=test_set, batch_size=64, shuffle=False, worker_init_fn=seed_worker, generator=g)

    return train_loader, test_loader



