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
        self.nn1 = nn.Linear(self.input_size, self.hidden_size)
        self.nn2 = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x: torch.tensor):
        '''Compute the forward pass.
        Args: x (torch.tensor): Input tensor
        Returns: torch.tensor: Output tensor
        '''
        x = torch.relu(self.nn1(x))
        x = self.nn2(x)
        return torch.softmax(x, dim=1)