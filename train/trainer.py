
import numpy as np
from util import utils, data_process, train_utils, data_util
import torch
import role.client as cli
from torch.utils.data import DataLoader
from abc import abstractmethod
from tqdm import tqdm
from role import cluster
import torch
class Trainer:
    def __init__(self, batch_size, min_second, max_second, lap, log_path, save_path, epochs, max_run, batch_amp,
                 sup_batch_size, eval_seed,cluster_limit):
        self.max_run = max_run
        self.batch_size = batch_size
        self.min_second = min_second
        self.max_second = max_second
        self.lap = lap
        self.save_path = save_path
        self.log_path = log_path
        self.epochs = epochs
        self.batch_amp = batch_amp
        self.sup_batch_size = sup_batch_size
        self.eval_seed = eval_seed
        self.cluster_limit = cluster_limit
        self.checkpoint = self.make_checkpoint()

        self.run_idx = None
        self.db = None
        # self.staff_user = None
        self.model_list = None
        self.optimizer_list = None
        self.start_epoch, self.record = None, None

        self.eval_loader_list = None
        self.eval_score = None
        self.staff_list = None
        # self.client_list = None
        self.sever = None
        self.clusters = None
    def make_checkpoint(self):
        return utils.CheckPoint(self.log_path)

    def start(self, **kwargs):
        if not self.check_run():
            return
        #处理原始数据
        self.db = self.create_database(**kwargs)

        while self.check_run():
            self.run_idx = self.create_read_run()
            #创建模型和优化器
            self.model_list = self.create_read_model_list()
            self.optimizer_list = self.create_read_optimizer_list()

            #读取训练记录中的start和record
            self.start_epoch, self.record = self.create_read_args()
            #创建数据加载器
            self.eval_loader_list = self.make_eval_loaders()
            self.sever = self.make_sever()
            #创建客户端列表
            self.staff_list = self.make_staffs()
            #创建聚类list
            self.clusters = self.create_clusters()
            #创建评分
            self.eval_score = self.make_score()
            self.before_train(kwargs['verbose'])
            self.train(verbose=kwargs['verbose'], show_idx=kwargs['show_idx'], log_idx=kwargs['log_idx'],
                       tqdm_verbose=kwargs['tqdm_verbose'] if 'tqdm_verbose' in kwargs.keys() else None)
            self.save()
    #检查是否运行过
    def check_run(self):
        run_idx = self.checkpoint.get('run_idx')

        return run_idx is None or run_idx < self.max_run

    def create_read_run(self):
        run_idx = self.checkpoint.get('run_idx')
        if run_idx is None:
            run_idx = 0
            self.checkpoint.update(run_idx=run_idx)
            self.checkpoint.write()

        return run_idx

    def make_score(self):
        return train_utils.Metric([
                train_utils.Metric_Item(train_utils.acc_score, name='eval_acc', average=False),
                train_utils.Metric_Item(train_utils.f1_score, name='f1', average='macro'),
            ])

    def sample_read_staff(self, **kwargs):
        staff_user = self.checkpoint.get('staff_user')
        if staff_user is None:
            if 'staff_user' in kwargs.keys() and kwargs['staff_user'][self.run_idx] is not None:
                staff_user = kwargs['staff_user'][self.run_idx]
            else:
                staff_user = [i for i in range(1,self.db.Uci.CLIENTS + 1)]
            self.checkpoint.update(staff_user=staff_user)

        return staff_user

    def create_read_args(self):
        if self.checkpoint.get('start') is None:
            record, start = [], 0
            self.checkpoint.update(record=record, start=start)
        else:
            start, record, = self.checkpoint.gets('start', 'record')

        return start, record

    @abstractmethod
    def create_read_model_list(self):
        pass

    def create_read_optimizer_list(self):
        optimizer_list = []
        #adam优化器，用于优化模型参数
        for model in self.model_list:
            optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.99), weight_decay=0.0005)
            optimizer_list.append(optimizer)
        if self.checkpoint.get('optimizer_list') is not None:
            optimizers = self.checkpoint.get('optimizer_list')
            for key,value in optimizers.items():
                optimizer_list[key].load_state_dict(value)
        return optimizer_list
    #创建数据库
    def make_eval_loaders(self):
        eval_loader_list = []
        #对于每个用户，创建一个数据加载器
        for idx in range(self.db.owner.user_count):
            eval_loader = DataLoader(
                data_util.ATTDataset(
                    #在这段代码中，`*` 符号用于解包 `self.db.owner.choose_mul` 方法返回的值。
                    # 这意味着，如果 `self.db.owner.choose_mul` 返回一个元组，那么这个元组中的每个元素都将作为单独的参数传递给 `data_util.ATTDataset` 类。例如，
                    # 如果 `self.db.owner.choose_mul` 返回 `(1, 2, 3)`，
                    # 那么这段代码相当于调用 `data_util.ATTDataset(1, 2, 3, seed=self.eval_seed)`
                    #对用户idx的每项活动进行采样
                    *self.db.owner.choose_mul(idx, {key: 600 for key in self.db.owner.activity_tag.keys()},
                                              seed=self.eval_seed)), batch_size=256)
            eval_loader_list.append(eval_loader)

        return eval_loader_list

    def make_staffs(self):
        staff_list = []
        for staff_idx in range(self.db.owner.user_count):
            staff = cli.Staff(self.db.owner, staff_idx, self.db.sup_samples, self.model_list[staff_idx], self.sup_batch_size)
            staff.train_init(seconds=self.db.sup_seconds, fresh=self.db.sup_fresh, lap=0.)
            staff_list.append(staff)

        return staff_list

    def eval(self):
        result_list = [train_utils.eval(self.model_list[user_idx], self.eval_loader_list[user_idx], self.eval_score)
                       for user_idx in range(self.db.owner.user_count)]
        #评估模式下的平均损失
        mean_value = {key: np.mean([result[key] for result in result_list]) for key in result_list[0].keys()}

        return result_list, mean_value

    def save(self):
        save = self.make_save_dict()
        print(save.keys())
        utils.write_pkl(self.save_path.format(self.run_idx), save)
        self.run_idx = self.run_idx + 1
        self.checkpoint.clear()
        self.checkpoint.update(run_idx=self.run_idx)
        self.checkpoint.write()

    def make_save_dict(self):
        return {'model_list': [model.cpu().state_dict() for model in  self.model_list], 'record': self.record}

    @abstractmethod
    def make_clients(self):
        pass

    @abstractmethod
    def make_sever(self):
        pass

    @abstractmethod
    def create_database(self, **kwargs):
        pass

    def before_train(self, verbose):
        pass

    def train(self, verbose, show_idx, log_idx, tqdm_verbose):
        epoch = range(self.start_epoch + 1, self.epochs + 1)
        epoch = tqdm(epoch) if verbose or tqdm_verbose else epoch
        for ep in epoch:
            flag = self.train_step(ep, verbose, show_idx, log_idx)
            if flag:
                break

    @abstractmethod
    def train_step(self, ep, verbose, show_idx, log_idx):
        pass


    def train_log(self, verbose, show_idx, log_idx, ep, **kwargs):
        if ep % (show_idx / 10) == 0:
            result_list, mean_value = self.eval()
            if verbose:
                for result in result_list:
                    train_utils.show_result(ep, result)
                train_utils.show_result(ep, '****mean: ', mean_value)
            self.record.append([ep, mean_value, result_list])

        if ep % (log_idx / 10) == 0:
            models = {}
            opentimizers = {}
            for index in self.db.owner.user_count:
                models[index] = self.model_list[index].state_dict()
                opentimizers[index] = self.optimizer_list[index].state_dict()
            self.checkpoint.update(start=ep, models=models, record=self.record,
                                   optimizers=opentimizers, **kwargs)
            self.checkpoint.write()
    def create_clusters(self):
        clusters = []
        for staff in self.staff_list:
            clusters.append(cluster.cluster([staff], [self.optimizer_list[staff.user_idx]]))
        return clusters
    def clusters_update(self,simility_metrix,cluster_limit):
        max_cosine_index = simility_metrix.unstack().idxmax()
        max_cosine_value = simility_metrix.unstack().max()
        if max_cosine_value > cluster_limit:
            self.clusters[max_cosine_index[0]].staff_list.append(self.clusters[max_cosine_index[1]].staff_list)
            self.clusters[max_cosine_index[0]].optimizer_list.append(self.clusters[max_cosine_index[1]].optimizer_list)
            self.clusters[max_cosine_index[0]].staff_gl.append(self.clusters[max_cosine_index[1]].staff_gl)
            self.clusters.remove(self.clusters[max_cosine_index[1]])
    def cos_similarities(self):
        pass