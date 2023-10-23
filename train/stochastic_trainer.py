import numpy as np
import pandas as pd
import util.utils
from util import utils, data_process, train_utils, data_util
from abc import abstractmethod
from train import trainer
import torch.nn.functional as F
import pandas as pd
import torch
import time as t
import role.cluster
class StochasticTrainer(trainer.Trainer):
    def __init__(self,**kwargs):
        trainer.Trainer.__init__(self,
            **{key: value for key, value in kwargs.items() if key in trainer.Trainer.__init__.__code__.co_varnames})



    def train_step(self, ep, verbose, show_idx, log_idx):
        staff_gl = []#存储所有cluster的梯度
        #对于每个cluster，计算梯度
        for cluster in self.clusters:
            self.sup_compute(cluster, verbose)
        #如果epoch大于4，计算相似度矩阵并更新cluster
        if ep > 4:
            simility_metric = self.cos_similarities()
            self.clusters_update(simility_metric,self.cluster_limit)
        #对于更新后的cluster,计算平均梯度
        for cluster in self.clusters:
            #对于有多个staff的cluster，计算平均梯度并更新模型
            self.sever.train_init(cluster.staff_gl)
            cluster.compute_grad_mean(self.sever)
            cluster.get_flat_grad()
            #将所有cluster的平均梯度存储到staff_gl中
            staff_gl.append(cluster.mean_gl[0])
        self.sever.train_init(staff_gl)
        #对于单独的staff，使用总的平均梯度计更新模型
        for cluster in self.clusters:
            if len(cluster.staff_list) == 1:
                self.avg_update(cluster)
            cluster.clear()
        if verbose:
            train_utils.show_result(ep, [staff.train_score.value()['loss'] for staff in self.staff_list])

        self.log_step(verbose, show_idx, log_idx, ep)

    @abstractmethod
    def sup_compute(self, cluster, metric):
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
    def cos_similarities (self):
        metrix = pd.DataFrame(data = np.zeros((len(self.clusters),len(self.clusters))),
                              columns=[f'cluster_{i}' for i in range(len(self.clusters))],
                              index=[f'cluster_{i}' for i in range(len(self.clusters))])
        for row in range(len(self.clusters)):
            for column in range(row + 1,len(self.clusters)):
                metrix.iloc[row,column] = F.cosine_similarity(self.clusters[row].flat_grad,
                                                              self.clusters[column].flat_grad,dim=0)
        return metrix