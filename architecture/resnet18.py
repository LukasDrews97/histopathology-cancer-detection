import torch
import torch.nn as nn
from torchvision.models.resnet import ResNet, BasicBlock

class Net(nn.Module):
    def __init__(self, img_dim) -> None:
        super().__init__()
        self.model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, img):
        img = self.model.forward(img)
        img = self.sigmoid(img)
        img = img.flatten()
        return img


