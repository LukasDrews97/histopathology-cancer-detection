import torch.nn as nn
from torchvision.models.densenet import DenseNet

class Net(nn.Module):
    """
    Class implementing the densenet121 model.
    """
    def __init__(self, img_dim) -> None:
        super().__init__()
        self.model = DenseNet(growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64, 
                        bn_size=4, drop_rate=0, num_classes=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, img):
        """
        Forward pass.
        Args:
            img:
                Images to calculate the forward pass.
        """
        img = self.model.forward(img)
        img = self.sigmoid(img)
        img = img.flatten()
        return img