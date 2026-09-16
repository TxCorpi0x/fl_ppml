import torch
from torch import nn
import torch.nn.functional as F


# Define the MLP architecture for tabular data (healthcare, etc.)
class TabularNet(nn.Module):
    """
    A simple MLP model for tabular data

    Args:
        input_dim: Number of input features
        num_classes: Number of output classes
        hidden_dims: List of hidden layer dimensions
    """

    def __init__(self, input_dim=13, num_classes=2, hidden_dims=[64, 32]) -> None:
        super(TabularNet, self).__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_classes))

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the neural network"""
        return self.model(x)


# Define the CNN architecture for image data
class Net(nn.Module):
    """
    A simple CNN model.

    Args:
        num_classes:  Number of output classes.
        in_channels:  Number of input channels (1 for grayscale, 3 for RGB).
        input_size:   Spatial size of the input image (assumed square, e.g. 28 or 32).
                      Used to compute the flattened dimension before the FC layers.
    """

    def __init__(self, num_classes=10, in_channels=3, input_size=32) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # Compute spatial size after two conv+pool steps: floor((S-4)/2) twice
        s1 = (input_size - 4) // 2  # after conv1(5) + pool(2)
        s2 = (s1 - 4) // 2  # after conv2(5) + pool(2)
        self._flat_dim = 16 * s2 * s2
        self.fc1 = nn.Linear(self._flat_dim, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the neural network
        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
