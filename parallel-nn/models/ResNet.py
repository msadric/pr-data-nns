import torch.nn as nn
import torch.nn.functional as F


class ResNet(nn.Module):
    """
    Residual Neural Network (ResNet) implementation.

    Args:
        block (nn.Module): The basic building block for the ResNet.
        layers (list): A list specifying the number of blocks in each layer.
        num_classes (int): The number of output classes.

    Attributes:
        in_channels (int): The number of input channels.
        conv1 (nn.Conv2d): The first convolutional layer.
        bn1 (nn.BatchNorm2d): Batch normalization layer after the first convolutional layer.
        relu (nn.ReLU): ReLU activation function.
        maxpool (nn.MaxPool2d): Max pooling layer.
        layer1 (nn.Sequential): The first layer of blocks.
        layer2 (nn.Sequential): The second layer of blocks.
        layer3 (nn.Sequential): The third layer of blocks.
        layer4 (nn.Sequential): The fourth layer of blocks.
        avgpool (nn.AdaptiveAvgPool2d): Adaptive average pooling layer.
        fc (nn.Linear): Fully connected layer for classification.

    """

    def __init__(self, block, layers, num_classes, input_channels=3):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.make_layer(block, 64, layers[0])
        self.layer2 = self.make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self.make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self.make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def make_layer(self, block, out_channels, blocks, stride=1):
        """
        Create a layer of blocks.

        Args:
            block (nn.Module): The basic building block for the layer.
            out_channels (int): The number of output channels for each block.
            blocks (int): The number of blocks in the layer.
            stride (int, optional): The stride for the first block. Defaults to 1.

        Returns:
            nn.Sequential: The layer of blocks.

        """
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass of the ResNet.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The logits and probabilities.

        """
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        logits = out
        probas = F.softmax(logits, dim=1)
        return logits, probas


def ResNet18(num_classes, input_channels=3):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, input_channels)


class BasicBlock(nn.Module):
    """
    Basic building block for the ResNet.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        stride (int, optional): The stride for the first convolutional layer. Defaults to 1.

    Attributes:
        conv1 (nn.Conv2d): The first convolutional layer.
        bn1 (nn.BatchNorm2d): Batch normalization layer after the first convolutional layer.
        relu (nn.ReLU): ReLU activation function.
        conv2 (nn.Conv2d): The second convolutional layer.
        bn2 (nn.BatchNorm2d): Batch normalization layer after the second convolutional layer.
        shortcut (nn.Sequential): The shortcut connection.

    """

    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        """
        Forward pass of the basic block.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.

        """
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out