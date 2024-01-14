import torch
import torch.nn as nn
from torch.nn import Module

class Attention(Module):

    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.weight = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Tanh(), nn.Softmax(dim=1))
        self.record = False
        self.record_value = None

    def forward(self, inputs :torch.Tensor):
        assert len(inputs.shape) == 3
        weight = self.weight(inputs)
        outputs = (inputs * weight).sum(dim=1)
        if self.record:
            if self.record_value is None:
                self.record_value = torch.tensor(weight.mean(dim=[0,2]).clone().detach().cpu().tolist())
            else:
                self.record_value = (torch.tensor(weight.mean(dim=[0,2]).clone().detach().cpu().tolist()) + self.record_value) / 2
        return outputs

    def record_on(self):
        self.record_value = None
        self.record = True

    def record_off(self):
        self.record_value = None
        self.record = False



