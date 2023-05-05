import torch
from torch import nn

class DNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(DNN, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self._init_layers()

    def _init_layers(self):
        self.nn1 = nn.Linear(self.input_size, self.hidden_size)
        self.nn2 = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        x = torch.relu(self.nn1(x))
        x = self.nn2(x)
        return torch.softmax(x, dim=1)