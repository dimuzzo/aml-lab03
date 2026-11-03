import torch
import torch.nn as nn

class CustomNet(nn.Module):
    """Custom Convolutional Neural Network architecture."""
    def __init__(self, num_classes=200):
        super(CustomNet, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)

        # Spatial dimension reduction
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Force the feature maps to have spatial size 8x8
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))

        # Fully connected classifier layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))

        x = self.adaptive_pool(x)
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x