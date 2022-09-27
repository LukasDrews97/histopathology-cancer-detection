import torch.nn as nn
from torchvision.models.densenet import DenseNet

class Net(nn.Module):
    """
    Class implementing the densenet121 model adepted for hyperparameter tuning.
    """
    def __init__(self, img_dim, growth_rate, num_init_features, drop_rate) -> None:
        super().__init__()
        self.model = DenseNet(growth_rate=growth_rate, block_config=(6, 12, 24, 16), num_init_features=num_init_features, 
                        bn_size=4, drop_rate=drop_rate, num_classes=1)
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