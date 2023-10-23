import train.trainer
import role.sever as server
import torch
from util.device import device
class cluster:
    def __init__(self, staff_list, optimizer_list):
        self.staff_list = staff_list
        self.optimizer_list = optimizer_list
        self.staff_gl = []
        self.mean_gl = []
        self.flat_grad = torch.tensor([], dtype=torch.float)
    def clear(self):
        self.staff_gl.clear()
        self.mean_gl.clear()
    def compute_grad_gl(self, metric):
        for staff in self.staff_list:
            gl_dict = staff.compute_grad(metric=metric)
            self.staff_gl.append(gl_dict)
    def compute_grad_mean(self,sever:server):
        sever.mean_update_cluster(self)
    def get_flat_grad(self):
        for mean_gl_dict in self.mean_gl:
            for name in list(mean_gl_dict.keys()):
                self.flat_grad = torch.cat((self.flat_grad, mean_gl_dict[name].flatten()), dim=0)
