import torch
from torch.utils.data import DataLoader
from dataset import HistopathologicImageDataset
from torchvision import transforms
from torch.utils.data import random_split


device = "cuda" if torch.cuda.is_available() else "cpu"
dataset = HistopathologicImageDataset(labels_file="./data/train_labels.csv", img_dir="./data/train/", transform=transforms.Compose([
    transforms.PILToTensor(),
    transforms.Resize([32,32]),
    transforms.Grayscale(num_output_channels=3),
    transforms.ConvertImageDtype(dtype=torch.float32),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
]))

train_set, test_set = random_split(dataset, [176_020, 44_005])
train_loader = DataLoader(dataset=train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=64, shuffle=True)


def main():
    print(len(dataset)) #220025 
    print(dataset[0][0].shape)


if __name__=="__main__": 
    main()