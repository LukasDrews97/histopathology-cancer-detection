import torch.nn as nn
from torchvision.models.densenet import DenseNet

class Net(nn.Module):
    def __init__(self, growth_rate, num_init_features, drop_rate) -> None:
        super().__init__()
        self.model = DenseNet(growth_rate=growth_rate, block_config=(6, 12, 24, 16), num_init_features=num_init_features, 
                        bn_size=4, drop_rate=drop_rate, num_classes=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, img):
        img = self.model.forward(img)
        img = self.sigmoid(img)
        img = img.flatten()
        return img