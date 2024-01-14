from util.device import device
import role.sever as sev
from train import database, stochastic_trainer
from model import att_model
import role.cluster

class StochasticNearTrainer(stochastic_trainer.StochasticTrainer):
    def __init__(self, semi=0.2, near_fn=None, **kwargs):
        stochastic_trainer.StochasticTrainer.__init__(self, **kwargs)
        self.semi = semi
        self.near_fn = near_fn

    def make_sever(self):
        sever = sev.NearSever(self.model)
        sever.train_init(self.optimizer)
        return sever
    #创建模型
    def create_read_model(self):
        model = att_model.UciATTModel(self.db.owner.class_num, self.db.owner.sensor_dim, self.db.owner.num_in_sensors,
                                       self.db.owner.devices, series_dropout=0.2, device_dropout=0.2, sensor_dropout=0.2,
                                       predict_dropout=0.4).float().to(device) #设置几个注意力层的dropout率，将数据转化为float类型并放到gpu上
        # #从checkpoint中读取预训练模型
        # if self.checkpoint.get('model_list') is not None:
        #     for idx in range(self.db.owner.user_count):
        #         model_list[idx].load_state_dict(self.checkpoint.get('model_list')[idx])

        return model

    def sup_compute(self, staff, metric):
        return staff.compute_grad(metric=metric)


    def avg_update(self,cluster:role.cluster):

        self.sever.mean_update_global(cluster)

    def log_step(self, verbose, show_idx, log_idx, ep):
        self.train_log(verbose, show_idx, log_idx, ep)


class UciStochasticNearTrainer(StochasticNearTrainer, database.Uci):
    def __init__(self, **kwargs):
        StochasticNearTrainer.__init__(self, **kwargs)
        database.Uci.__init__(self, **kwargs)

    def create_database(self, **kwargs):

        self.make_owner()
        return self

