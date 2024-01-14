import pandas as pd
from util import utils, data_process, train_utils, data_util
from abc import abstractmethod
from train import trainer
import torch.nn.functional as F
import torch
import role.cluster
clip = 0.08
class StochasticTrainer(trainer.Trainer):
    def __init__(self,**kwargs):
        trainer.Trainer.__init__(self,
            **{key: value for key, value in kwargs.items() if key in trainer.Trainer.__init__.__code__.co_varnames})


    #fedavg
    def train_step(self, ep, verbose, show_idx, log_idx):
        staff_gl = []#存储所有cluster的梯度
        for staff in self.staff_list:
            grad_dict = self.sup_compute(staff,verbose)
            staff_gl.append(grad_dict)
        # self.clusters_update(self.cluster_limit,max_cosine_value,max_cosine_index)
        # #对于更新后的cluster,计算平均梯度
        # for cluster in self.clusters:
        #     #对于有多个staff的cluster，计算平均梯度并更新模型
        #     self.sever.train_init(cluster.staff_gl)
        #     cluster.compute_grad_mean(self.sever)
        #     cluster.get_flat_grad()
        #     #将所有cluster的平均梯度存储到staff_gl中
        #     staff_gl.append(cluster.mean_gl[0])
        # self.sever.train_init(staff_gl)
        # #对于单独的staff，使用总的平均梯度计更新模型
        # for cluster in self.clusters:
        #     if len(cluster.staff_list) == 1:
        #         self.avg_update(cluster)
        #     cluster.clear()
        if verbose:
            train_utils.show_result(ep, [staff.train_score.value()['loss'] for staff in self.staff_list])
        self.sever.staff_gl += staff_gl
        self.sever.mean_update()
        self.log_step(verbose, show_idx, log_idx, ep)

    @abstractmethod
    def sup_compute(self,staff, metric) -> dict:
        pass

    @abstractmethod
    def semi_compute(self, client, metric) -> dict:
        pass

    @abstractmethod
    def avg_update(self, cluster:role.cluster):
        pass

    @abstractmethod
    def log_step(self, verbose, show_idx, log_idx, ep):
        pass
