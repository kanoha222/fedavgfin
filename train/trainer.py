
import numpy as np
from util import utils, data_process, train_utils, data_util
import torch
import role.client as cli
from torch.utils.data import DataLoader
from abc import abstractmethod
from tqdm import tqdm
from role import cluster
import torch
import wandb
from util.rdp_analysis import calibrating_sampled_gaussian
q = 0.08
eps = 4
bad_envent = 1e-5

class Trainer:
    def __init__(self,lap, log_path, save_path, epochs, max_run,
                 sup_batch_size, eval_seed,cluster_limit):
        self.max_run = max_run
        self.lap = lap
        self.save_path = save_path
        self.log_path = log_path
        self.epochs = epochs
        self.sup_batch_size = sup_batch_size
        self.eval_seed = eval_seed
        self.cluster_limit = cluster_limit
        #checkpoint用于保存run_idx
        self.checkpoint = self.make_checkpoint()
        self.loadpoint = self.make_loadpoint()
        self.run_idx = None
        self.db = None
        self.model = None
        self.optimizer = None
        self.start_epoch, self.record = None, None
        self.eval_loader_list = None
        self.eval_score = None
        self.staff_list = None
        self.sever = None
        # self.clusters = None
        self.wandb = None
        self.sigma = None
    def make_checkpoint(self):
        return utils.Point(self.log_path)
    def make_loadpoint(self):
        return utils.Point(self.save_path)
    def start(self, **kwargs):
        if not self.check_run():
            return
        #处理原始数据
        self.db = self.create_database(**kwargs)

        while self.check_run():
            self.run_idx = self.create_read_run()
            #创建模型和优化器
            self.model = self.create_read_model()
            self.optimizer = self.create_read_optimizer()

            #读取训练记录中的start和record
            self.start_epoch, self.record = self.create_read_args()
            #创建数据加载器
            self.eval_loader_list = self.make_eval_loaders()
            self.sever = self.make_sever()
            #创建客户端列表
            self.staff_list = self.make_staffs()
            self.wandb = wandb.init( project=f'myfedhar',
                                     group='fedavg',
                                     name=f'test1',
                                     config={
                                             'optimizer': 'Adam',
                                             'learning_rate': 0.001} )

            #创建聚类list
            # self.clusters = self.create_read_clusters()
            #创建评分
            self.eval_score = self.make_score()
            # self.sigma = calibrating_sampled_gaussian(q,eps, bad_envent,iters = self.epochs)
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
    def create_read_model(self):
        pass

    def create_read_optimizer(self):
        # optimizer_list = []
        optimizer = torch.optim.Adam(self.model.parameters(), betas=(0.9, 0.99), weight_decay=0.0005,lr=0.001)
        return optimizer
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
        if self.loadpoint.get('staff_list') is not None:
            staff_list = self.loadpoint.get('staff_list')
        else:
            for staff_idx in range(self.db.owner.user_count):
                staff = cli.Staff(self.db.owner, staff_idx, self.db.sup_samples, self.model, self.sup_batch_size)
                staff.train_init(seconds=self.db.sup_seconds, fresh=self.db.sup_fresh, lap=0.)
                staff_list.append(staff)

        return staff_list

    def eval(self):
        result_list = [train_utils.eval(self.model, self.eval_loader_list[user_idx], self.eval_score)
                       for user_idx in range(self.db.owner.user_count)]
        #评估模式下的平均损失
        mean_value = {key: np.mean([result[key] for result in result_list]) for key in result_list[0].keys()}

        return result_list, mean_value

    def save(self):
        save = self.make_save_dict()
        utils.write_pkl(self.save_path.format(self.run_idx), save)
        self.run_idx = self.run_idx + 1
        self.checkpoint.clear()
        self.checkpoint.update(run_idx=self.run_idx)
        self.checkpoint.write()

    def make_save_dict(self):
        # clusters_show = []
        # i = 0
        # for cluster in self.clusters:
        #     cluster_dict = {f'cluster_{i}': [staff.user_idx for staff in cluster.staff_list]}
        #     clusters_show.append(cluster_dict)
        #     i += 1
        return {'model': self.model.cpu().state_dict(), 'record': self.record}


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
                # clusters_show = []
                # if ep > 4:
                #     i = 0
                #     for cluster in self.clusters:
                #         cluster_dict = {f'cluster_{i}': [staff.user_idx for staff in cluster.staff_list] }
                #         clusters_show.append(cluster_dict)
                #         i += 1
                train_utils.show_result(ep, '****mean: ', mean_value,'\n'
                                        # ,clusters_show if clusters_show is not None else None
                                        )

            all_loss = {f'staff_{user_idx}_loss': result_list[user_idx]['eval_loss'] for user_idx in range(self.db.owner.user_count)}
            all_acc = {f'staff_{user_idx}_acc': result_list[user_idx]['eval_acc'] for user_idx in range(self.db.owner.user_count)}
            all_f1 = {f'staff_{user_idx}_f1': result_list[user_idx]['f1'] for user_idx in range(self.db.owner.user_count)}
            self.wandb.log({**all_loss,**all_acc,**all_f1})
            self.record.append([ep, mean_value, result_list])

        if ep % (log_idx / 10) == 0:
            self.checkpoint.update(start=ep, model=self.model.state_dict(), record=self.record,
                                   optimizer=self.optimizer.state_dict(), **kwargs)
            self.checkpoint.write()
