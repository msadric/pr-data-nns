"""AlexNet model class."""
import torch
import torch.nn.functional as F



# MODEL
# Define neural network by subclassing PyTorch's nn.Module.
class AlexNet(torch.nn.Module):
    """
    AlexNet model for image classification.

    Attributes
    ----------
    features : torch.nn.container.Sequential
        The convolutional feature-extractor part of AlexNet.
    avgpool : AdaptiveAvgPool2d
        An adaptive pooling layer.
    classifier : torch.nn.container.Sequential
        The fully connected linear part of AlexNet.

    Methods
    -------
    forward()
        The forward pass.
    """

    def __init__(self, num_classes: int = 1000, dropout: float = 0.5, input_channels=3) -> None:
        """
        Initialize AlexNet architecture.

        Parameters
        ----------
        num_classes : int
            The number of classes in the underlying classification problem.
        dropout : float
            The dropout probability.
        """
        super().__init__()
        self.features = torch.nn.Sequential(
            # AlexNet consists of 8 layers:
            # 5 convolutional layers, some followed by max-pooling,
            # and 3 fully connected layers.
            # IMPLEMENT FEATURE-EXTRACTOR PART OF ALEXNET HERE!
            # 1st convolutional layer (+ max-pooling)
            torch.nn.Conv2d(input_channels, 64, kernel_size=11, stride=4, padding=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
            # 2nd convolutional layer (+ max-pooling)
            torch.nn.Conv2d(64, 192, kernel_size=5, padding=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
            # 3rd + 4th convolutional layer
            torch.nn.Conv2d(192, 384, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(384, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            # 5th convolutional layer
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
        )
        # Average pooling to downscale possibly larger input images.
        self.avgpool = torch.nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = torch.nn.Sequential(
            # IMPLEMENT FULLY CONNECTED MULTI-LAYER PERCEPTRON PART HERE!
            # 6th, 7th + 8th fully connected layer
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(256 * 6 * 6, 4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(4096, num_classes),
            ###################################
        )

    # Every nn.Module subclass implements the operations on the input data in the forward method.
    # Forward pass: Apply AlexNet model to input x.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The forward pass.

        Parameters
        ----------
        x : torch.Tensor
            The input data.

        Returns
        -------
        torch.Tensor
            The output.
        """
        # IMPLEMENT OPERATIONS ON INPUT DATA x HERE!
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        logits = x
        probas = F.softmax(logits, dim=1)
        return logits, probas
