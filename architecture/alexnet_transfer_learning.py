import torch.nn as nn
from torchvision.models import alexnet
from torchvision.models.alexnet import AlexNet_Weights

class Net(nn.Module):
    """
    Class implementing the transfer learning model using AlexNet.
    """
    def __init__(self, img_dim):
        super().__init__()

        #Instatiate model
        self.model = alexnet(weights=AlexNet_Weights.DEFAULT)

        #Now we adapt the AlexNet:
        self.model.classifier[1] = nn.Linear(9216, 1024)

        #Updating the second classifier
        self.model.classifier[4] = nn.Linear(1024, 256)

        #Updating the third and last classifier that is the output layer of the network.
        self.model.classifier[6] = nn.Linear(256, 1)       

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