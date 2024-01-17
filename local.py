import numpy as np

import train.trainer
from model import att_model
from util import data_process, train_utils, data_util, utils
from util.device import device,to_device
import util.Path as Path
import torch
import model.model as mod
from torch.utils.data import DataLoader
freq_rate = 1.
time_window = int(2.5 * 1000)
allow_rate = 0.8
sup_fresh = 200
sup_seconds = 600
sup_batch_size = 128
eval_seed = 4
from tqdm import tqdm
class Client:
    def __init__(self, owner: data_process.DataOwner, model: mod.ATTModel, batch_size=256):
        self.owner = owner
        self.model = model
        self.batch_size = batch_size
        self.is_local = False

    def train_init(self, *args, **kwargs):
        pass


    def compute_grad(self, *args, **kwargs) -> dict:
        pass


class Staff(Client):
    def __init__(self, owner: data_process.DataOwner, user_idx, samples, model: mod.ATTModel, batch_size):
        super(Staff, self).__init__(owner, model, batch_size)
        self.user_idx = user_idx
        self.samples = samples
        self.action_tup = self.owner.get_time_tup(user_idx, samples)
    def train_init(self, seconds=250, fresh=3, lap=0.):
        self.generator = self.generate(seconds, fresh, lap)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.train_score = train_utils.Metric()


    def generate(self, seconds, fresh, lap):
        while True:
            xs, ys = self.owner.choose_mul(self.user_idx, {action: seconds for action in self.action_tup.keys()},
                                           action_tup=self.action_tup, lap=lap)
            #xs和ys表示用户user_idx的所有活动的采樣数据
            for idx in range(fresh):
                loader = data_util.MLoader(
                    DataLoader(data_util.ATTDataset(xs, ys), batch_size=self.batch_size, shuffle=True), loop=False)
                #按照batch_size大小，迭代数据集
                xys = next(loader)
                while xys is not None:
                    yield xys
                    xys = next(loader)

    def compute_grad(self,*args, **kwargs):
        self.model = self.model.to(device)
        self.model.train()
        xs, ys = next(self.generator)
        ys = to_device(ys)
        xs = to_device(xs)

        pred = self.model(xs)

        ce_loss = self.loss_fn(pred, ys)
        loss = ce_loss + self.model.regular()

        if kwargs['metric']:
            self.train_score.step(loss={'loss': loss})

        loss.backward()
        #该方法进行梯度剪裁，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 100, norm_type=2)

        # grad_dict = {}

        self.model.zero_grad()

        # return grad_dict
db = data_process.FftDataOwner(path=Path.online_uci.format('all_all'),
                                freq_rate=freq_rate,
                                time_window=time_window,
                                allow_rate=allow_rate,
                                expand_last=True)

def create_read_model():
    model = att_model.UciATTModel(db.class_num, db.sensor_dim, db.num_in_sensors,
                                   db.devices, series_dropout=0.2, device_dropout=0.2, sensor_dropout=0.2,
                                   predict_dropout=0.4).float().to(device) #设置几个注意力层的dropout率，将数据转化为float类型并放到gpu上

    return model
def create_read_optimizer(model):
    # optimizer_list = []
    optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.99), weight_decay=0.0005,lr=0.001)
    return optimizer

def eval(model):
    result = [train_utils.eval( model=model,
                                eval_loader=DataLoader(data_util.ATTDataset(*db.choose_mul(0, {key: 600 for key in db.activity_tag.keys()},seed=eval_seed)), batch_size=256)                              ,
                                score=train_utils.Metric([
                                train_utils.Metric_Item(train_utils.acc_score, name='eval_acc', average=False),
                                train_utils.Metric_Item(train_utils.f1_score, name='f1', average='macro')
            ]))
            ]

    return result
if __name__ == '__main__':

    model = create_read_model()
    optimizer = create_read_optimizer(model)
    staff = Staff(db, 0, 100, model, sup_batch_size)
    staff.train_init(seconds=sup_seconds, fresh=sup_fresh, lap=0.)
    epoch = range(1, 1000)
    epoch = tqdm(epoch)
    record = []
    for i in epoch:
        staff.compute_grad(metric=True)
        optimizer.step()
        optimizer.zero_grad()
        if 1:
            train_utils.show_result(i, [staff.train_score.value()['loss']])
        result = eval(model)
        train_utils.show_result(f'{i}', result)
        record.append([i, result])

    utils.write_pkl((Path.feder.format(dataset='uci',
                                    model="stochastic_near", name='local')), record)


