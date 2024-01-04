import torch
from torch import nn


class DNN(nn.Module):
    """DNN Class for the snake game."""

    def __init__(self, output_size: int, hidden_size: int):
        """Initialize the DNN.
        Args: input_size (int): Input size
              output_size (int): Output size
              hidden_size (int): Hidden size
        """
        super(DNN, self).__init__()
        self.input_size = 12
        self.output_size = output_size
        self.hidden_size = hidden_size
        self._init_layers()

    def _init_layers(self):
        """Initialize the layers."""
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc4 = nn.Linear(self.hidden_size, self.output_size)

    def forward(
        self,
        direction: torch.tensor,
        food_distance: torch.tensor,
        snake_body_danger: torch.tensor,
        snake_wall_danger: torch.tensor,
    ) -> torch.tensor:
        """Compute the forward pass.
        Args: direction (torch.tensor): Direction
              food_distance (torch.tensor): Food distance
              snake_body_danger (torch.tensor): Snake body danger
              snake_wall_danger (torch.tensor): Snake wall danger
        Returns: torch.tensor: Output
        """
        x = torch.cat(
            [direction, food_distance, snake_body_danger, snake_wall_danger], dim=1
        )
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)

        return x
