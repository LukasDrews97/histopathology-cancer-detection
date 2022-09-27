#https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files

import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image

class HistopathologicImageDataset(Dataset):
    """
    Creates a class respresenting the image dataset.
    """
    def __init__(self, labels_file, img_dir, transform=None, target_transform=None):
        """
        Initialize the class.
        Args:
            labels_file:
                Path to the file containing the image labels.
            img_dir:
                Path to the directory containing the images.
            transform:
                List of torchvision image transforms.
            target_transform:
                List of torchvision label transforms.
        
        """
        self.img_labels = pd.read_csv(labels_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        """
        Returns the length of the dataset.
        """
        return len(self.img_labels)

    def __getitem__(self, idx):
        """
        Method to retrieve items using an index. Overrides the [] operation.
        Args:
            idx:
                The integer index.
        """
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(f"{img_path}.tif")
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label