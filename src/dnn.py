import torch
from torch import nn

class DNN(nn.Module):
    '''DNN Class for the snake game.'''
    def __init__(self, output_size: int, hidden_size: int):
        '''Initialize the DNN.
        Args: input_size (int): Input size
              output_size (int): Output size
              hidden_size (int): Hidden size
        '''
        super(DNN, self).__init__()
        self.input_size = 8
        self.output_size = output_size
        self.hidden_size = hidden_size
        self._init_layers()

    def _init_layers(self):
        '''Initialize the layers.'''
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.lstm = nn.LSTM(2, self.hidden_size, batch_first=True)
        self.fc2 = nn.Linear(self.hidden_size*2, self.output_size)

    def forward(self, direction: torch.tensor,
                food_distance: torch.tensor,
                board_limits_distance: torch.tensor,
                snake_body: torch.tensor,
                ) -> torch.tensor:
        '''Compute the forward pass.
        Args: direction (torch.tensor): Direction
              food_distance (torch.tensor): Food distance
              board_limits_distance (torch.tensor): Board limits distance
              snake_body (torch.tensor): Snake body
        Returns: torch.tensor: Output
        '''
        x = torch.cat([direction, food_distance, board_limits_distance], dim=1)
        x = torch.relu(self.fc1(x))
        _, [y,_] = self.lstm(snake_body)
        x = self.fc2(torch.cat([x, y.squeeze(1)], dim=1))

        return torch.softmax(x, dim=1)