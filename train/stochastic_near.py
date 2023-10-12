from util.device import device
import role.sever as sev
from train import database, stochastic_trainer
from model import att_model


class StochasticNearTrainer(stochastic_trainer.StochasticTrainer):
    def __init__(self, semi=0.2, near_fn=None, **kwargs):
        stochastic_trainer.StochasticTrainer.__init__(self, **kwargs)
        self.semi = semi
        self.near_fn = near_fn

    # def make_clients(self):
    #     client_list = []
    #     for user_idx in range(self.db.owner.user_count):
    #         if user_idx not in self.staff_user:
    #             client = onc.OnlineSemiClient(self.db.owner, user_idx, self.model,
    #                                           batch_size=self.batch_size * self.batch_amp, near_fn=self.near_fn)
    #             client.train_init(self.min_second, self.max_second, self.lap, self.db.semi_seconds, self.db.semi_fresh)
    #             client_list.append(client)
    #
    #     return client_list

    def make_sever(self):
        sever = sev.NearSever()
        sever.train_init()

        return sever
    #创建模型
    def create_read_model_list(self):
        model = att_model.UciATTModel(self.db.owner.class_num, self.db.owner.sensor_dim, self.db.owner.num_in_sensors,
                                       self.db.owner.devices, series_dropout=0.2, device_dropout=0.2, sensor_dropout=0.2,
                                       predict_dropout=0.4).float().to(device) #设置几个注意力层的dropout率，将数据转化为float类型并放到gpu上
        self.model_list = [model] * self.db.owner.user_count
        #从checkpoint中读取预训练模型
        if self.checkpoint.get('model_list') is not None:
            for idx in range(self.db.owner.user_count):
                self.model_list[idx].load_state_dict(self.checkpoint.get('model_list')[idx])

        return self.model_list

    def sup_compute(self, cluster, metric):
        return cluster.compute_grad_gl(metric=metric)

    # def semi_compute(self, client, metric):
    #     return client.compute_grad(metric=metric)

    def avg_update(self):

        self.sever.mean_update()

    def log_step(self, verbose, show_idx, log_idx, ep):
        self.train_log(verbose, show_idx, log_idx, ep)


class UciStochasticNearTrainer(StochasticNearTrainer, database.Uci):
    def __init__(self, **kwargs):
        StochasticNearTrainer.__init__(self, **kwargs)
        database.Uci.__init__(self, **kwargs)

    def create_database(self, **kwargs):
        self.make_owner()
        return self

