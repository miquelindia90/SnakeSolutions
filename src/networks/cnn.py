import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import sys


def _get_CNN_ax_output_dimension(inputDimension):
    outputDimension = np.ceil(np.array(inputDimension, dtype=np.float32) / 2)
    outputDimension = np.ceil(np.array(outputDimension, dtype=np.float32) / 2)
    outputDimension = np.ceil(np.array(outputDimension, dtype=np.float32) / 2)
    return int(outputDimension)


class CNN(nn.Module):
    """CNN Class for the snake game."""

    def __init__(self, surface_dimensions: tuple[int, int], output_size: int, hidden_size: int):
        """Initialize the DNN.
        Args: output_size (int): Output size
              hidden_size (int): Hidden size
        """
        super(CNN, self).__init__()
        self.output_size = output_size
        self.cnn_output_size = int(32*_get_CNN_ax_output_dimension(surface_dimensions[0])*_get_CNN_ax_output_dimension(surface_dimensions[1]))
        self.hidden_size = hidden_size
        self._init_layers()

    def _init_layers(self):
        """Initialize the layers."""
        self.conv1 = nn.Conv2d(2, 8, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.activation = nn.ReLU()
        self.hidden_layer = nn.Linear(self.cnn_output_size, self.hidden_size)
        self.output_layer = nn.Linear(self.hidden_size, self.output_size)

    def forward(
        self,
        direction: torch.tensor,
        food_distance: torch.tensor,
        snake_body_danger: torch.tensor,
        snake_wall_danger: torch.tensor,
        board_tensor: torch.tensor,
    ) -> torch.tensor:
        """Compute the forward pass.
        Args: direction (torch.tensor): Direction
              food_distance (torch.tensor): Food distance
              snake_body_danger (torch.tensor): Snake body danger
              snake_wall_danger (torch.tensor): Snake wall danger
        Returns: torch.tensor: Output
        """
        x = F.max_pool2d(self.activation(self.bn1(self.conv1(board_tensor))), 2, stride=2, ceil_mode=True)
        x = F.max_pool2d(self.activation(self.bn2(self.conv2(x))), 2, stride=2, ceil_mode=True)
        x = F.max_pool2d(self.activation(self.bn3(self.conv3(x))), 2, stride=2, ceil_mode=True)
        x = x.contiguous().view(x.size()[0], self.cnn_output_size)
        x = self.activation(self.hidden_layer(x))
        x  = self.output_layer(x)
        return x
