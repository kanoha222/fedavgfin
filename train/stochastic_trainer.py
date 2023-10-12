import numpy as np

import util.utils
from util import utils, data_process, train_utils, data_util
from abc import abstractmethod
from train import trainer

class StochasticTrainer(trainer.Trainer):
    def __init__(self,**kwargs):
        trainer.Trainer.__init__(self,
            **{key: value for key, value in kwargs.items() if key in trainer.Trainer.__init__.__code__.co_varnames})



    def train_step(self, ep, verbose, show_idx, log_idx):
        staff_gl = []#存储所有cluster的梯度
        for cluster in self.clusters:
            staff_gl.append(self.sup_compute(cluster, verbose))
        if ep > 4:
            simility_metric = util.utils.cos_similarities(staff_gl)
            self.clusters_update(simility_metric)



        if verbose:
            train_utils.show_result(ep, [staff.train_score.value()['loss'] for staff in self.staff_list])
        self.sever.staff_gl += staff_gl

        # users = np.random.choice(len(self.client_list), size=self.sto_num, replace=False)
        # for _ in range(self.agg):
        #     client_gl = []
        #     for idx in users:
        #         grad_dict = self.semi_compute(self.client_list[idx], metric=verbose)
        #         client_gl.append(grad_dict)
        #     if verbose:
        #         train_utils.show_result(ep, [self.client_list[idx].train_score.value()['loss'] for idx in users])
        #     self.sever.client_gl += client_gl

        self.avg_update()
        self.log_step(verbose, show_idx, log_idx, ep)

    @abstractmethod
    def sup_compute(self, cluster, metric) -> dict:
        pass

    @abstractmethod
    def semi_compute(self, client, metric) -> dict:
        pass

    @abstractmethod
    def avg_update(self):
        pass

    @abstractmethod
    def log_step(self, verbose, show_idx, log_idx, ep):
        pass
