import torch
import torch.nn as nn
from util.device import device
from typing import List, Tuple
from util import train_utils
import role.client as client
from abc import abstractmethod
from copy import deepcopy

class Sever:
    def __init__(self, model: nn.Module):
        self.model = model
        self.staff_gl = []

    def train_init(self, optimizer):
        self.optimizer = optimizer

    @abstractmethod
    def mean_update(self, *args, **kwargs):
        pass

    def clear(self):
        self.staff_gl.clear()


class NearSever(Sever):
    def __init__(self, *args, **kwargs):
        super(NearSever, self).__init__(*args, **kwargs)
    #
    def mean_update(self):
        #将模型参数转移到GPU上
        self.model = self.model.to(device)
        #对于每个参数，计算所有客户端的梯度的平均值
        for name, param in self.model.named_parameters():
            staff_grad = torch.mean(torch.stack([grad_dict[name] for grad_dict in self.staff_gl], dim=0), dim=0)
            # client_grad = torch.mean(torch.stack([grad_dict[name] for grad_dict in self.client_gl], dim=0), dim=0)
            param.grad = staff_grad.to(device)
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.clear()

# class SuperSever(Sever):
#     def __init__(self, *args, **kwargs):
#         super(SuperSever, self).__init__(*args, **kwargs)
#
#     def mean_update(self):
#         self.model = self.model.to(device)
#         for name, param in self.model.named_parameters():
#             staff_grad = torch.mean(torch.stack([grad_dict[name] for grad_dict in self.staff_gl], dim=0), dim=0)
#             param.grad = staff_grad.to(device)
#         self.optimizer.step()
#         self.optimizer.zero_grad()
#         self.clear()
