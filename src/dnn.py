import torch
from torch import nn

class DNN(nn.Module):
    '''DNN Class for the snake game.'''
    def __init__(self, input_size: int, output_size: int, hidden_size: int):
        '''Initialize the DNN.
        Args: input_size (int): Input size
              output_size (int): Output size
              hidden_size (int): Hidden size
        '''
        super(DNN, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self._init_layers()

    def _init_layers(self):
        '''Initialize the layers.'''
        self.fc1 = nn.Linear(6, self.hidden_size)
        self.lstm = nn.LSTM(2, self.hidden_size, batch_first=True)
        self.fc2 = nn.Linear(self.hidden_size*2, self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, observation_space: torch.tensor,
                food_distance: torch.tensor,
                board_limits_distance: torch.tensor,
                snake_body: torch.tensor,
                ) -> torch.tensor:
        '''Compute the forward pass.
        Args: observation_space (torch.tensor): Observation space
              food_distance (torch.tensor): Distance from the snake head to the food
        Returns: torch.tensor: DNN logits
        '''
        x = torch.cat([food_distance, board_limits_distance], dim=1)
        x = torch.relu(self.fc1(x))
        _, [y,_] = self.lstm(snake_body)
        x = torch.relu(self.fc2(torch.cat([x, y.squeeze(1)], dim=1)))
        x = self.fc3(x)

        return torch.softmax(x, dim=1)