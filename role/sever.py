import torch
import torch.nn as nn
from util.device import device
from typing import List, Tuple
from util import train_utils
import role.client as client
from abc import abstractmethod
from copy import deepcopy
import role.cluster
class Sever:
    def __init__(self):
        self.staff_gl = []
        self.mean_gl = []
    def train_init(self,staff_gl):
        self.staff_gl = staff_gl
        name = list(staff_gl[0].keys() if staff_gl is not None else None)
        if len(name) >= 1:
            for str in name:
                param_list = []
                for list_gl in self.staff_gl:
                    param_list.append(list_gl[str])
                self.mean_gl.append({f'{str}':torch.mean(torch.stack(param_list,dim=0), dim=0)})

    @abstractmethod
    def mean_update(self, *args, **kwargs):
        pass

    def clear(self):
        self.staff_gl = []

class NearSever(Sever):
    def __init__(self, *args, **kwargs):
        super(NearSever, self).__init__()
    #
    def mean_update_cluster(self,cluster:role.cluster):
        if len(cluster.staff_list) > 1:
            for staff,optimizer in zip(cluster.staff_list,cluster.optimizer_list):
                index = 0
                for (name, param) in staff.model.named_parameters():
                    mean_grad = torch.tensor(self.mean_gl[index][name])
                    param.grad = mean_grad.to(device)
                    index += 1

                optimizer.step()
                optimizer.zero_grad()
            cluster.mean_gl = self.mean_gl
        else:
           cluster.mean_gl = cluster.staff_gl
        self.clear()
    def mean_update_global(self,cluster:role.cluster):
        index = 0
        for (name, param) in cluster.staff_list[0].model.named_parameters():
            mean_grad = torch.tensor(self.mean_gl[index][name])
            param.grad = mean_grad.to(device)
            index += 1
        cluster.optimizer_list[0].step()
        cluster.optimizer_list[0].zero_grad()
        self.clear()

