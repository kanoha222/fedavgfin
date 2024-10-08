from typing import List
from sklearn.preprocessing import StandardScaler
import numpy as np
from util import data_util, utils
import torch
import warnings
from abc import abstractmethod

class DataOwner:
    def __init__(self, path: str, time_window):
        self.userset, self.userlen, self.activity_tag, self.sensor_freq, self.sensor_dim, self.devices = utils.read_pkl(path)
        self.user_count = len(self.userset)
        self.class_num = len(list(self.activity_tag.keys()))
        #time_window = 2500
        self.time_window = time_window
        #num_in_sensors是一个字典，key是传感器，value是该传感器在一个时间窗口内的采样点数
        self.num_in_sensors = {sensor: int(freq * self.time_window/1000) for sensor, freq in self.sensor_freq.items()}

    def get_len_in_users(self):
        user_action_sec = []
        for user_idx in range(self.user_count):
            action_sec = {}
            for action in self.activity_tag.keys():
                time_tup = self.userlen[user_idx][action]
                sec = time_tup[1] - time_tup[0]
                action_sec[action] = [sec//1000, sec // self.time_window]
            user_action_sec.append(action_sec)

        mean_sum_sec = np.mean(np.sum([list(action_sec.values()) for action_sec in user_action_sec], axis=1), axis=0)

        return user_action_sec, mean_sum_sec

    def get_time_tup(self, user_idx, sample_num: int):
        action_tup = {}
        seconds = sample_num * self.time_window
        action_sec = {action: value[1] - value[0] for action, value in self.userlen[user_idx].items()}

        is_add = True
        while is_add:
            is_add = False
            mean_sec = seconds // (self.class_num - len(action_tup.keys()))
            for action in self.activity_tag.keys():
                if action not in action_tup.keys() and action_sec[action] <= mean_sec:
                    action_tup[action] = self.userlen[user_idx][action]
                    seconds = seconds - action_sec[action]
                    is_add = True
            if self.class_num == len(action_tup.keys()):
                break

        if seconds > 0 and len(action_tup.keys()) < self.class_num:
            mean_sec = seconds // (self.class_num - len(action_tup.keys()))
            for action in self.activity_tag.keys():
                if action not in action_tup.keys():
                    tup = self.userlen[user_idx][action]
                    action_tup[action] = (tup[0], tup[0]+mean_sec)

        return action_tup

    @abstractmethod
    def choose_in_one_loop(self, user_idx, action, seconds, start_time, lap, time_tup):
        pass

    def choose(self, user_idx, action: str, seconds: int, lap=0., seed=None, start_time=None, action_tup=None):
        #time_tup：某个用户的某个动作的开始时间和结束时间
        time_tup = self.userlen[user_idx][action] if action_tup is None else action_tup[action]
        max_len = (time_tup[1] - time_tup[0]) // 1000

        if start_time is None:
            if seed is not None:
                np.random.seed(seed)
            #在一个时间窗口内随机选择一个时间点作为开始时间
            start_time = np.random.choice(range(time_tup[0], time_tup[1], 1), size=1)[0]
        else:
            if start_time == -1:
                start_time = time_tup[0]
        #st_iter是一个生成器，用于生成一个时间窗口内的采样点
        st_iter = self.dicho(start_time, start_time + self.time_window*1000)
        xs, ys = [], []
        #最大采样点数
        row_num = self.seconds_to_samples(seconds)
        #当row_num大于该动作的采样点数时，就将该动作的采样点数添加到xs和ys中
        while row_num > len(ys):
            #next(st_iter)是一个整数，表示一个采样点的时间
            start_time = int(next(st_iter))
            #choose_in_one_loop是一个抽象方法，用于选择一个时间窗口内的采样点
            x, y = self.choose_in_one_loop(user_idx, action, max_len, start_time, lap, time_tup)
            xs += x[:row_num-len(ys)]
            ys += y[:row_num-len(ys)]

        return xs, ys

    def seconds_to_samples(self, seconds):
        return int(seconds * 1000 / self.time_window)

    def dicho(self, a, b):
        ml = [(a, b)]
        yield a
        while len(ml) > 0:
            x, y = ml.pop(0)
            mean = (x + y) / 2
            ml.append((x, mean))
            ml.append((mean, y))
            yield mean
    #该方法用于将数据集中的数据按照时间窗口进行切分
    def choose_mul(self, user_idx, action_seconds, seed=None, start_time=None, action_tup=None, lap=0., return_dict=False):
        result = {}
        #action_seconds是一个字典，key是动作，value是该动作的时间长度
        for action, seconds in action_seconds.items():
            #time_tup是一个元组，元组中的两个元素分别是该动作的开始时间和结束时间
            time_tup = self.userlen[user_idx][action]
            #如果seconds为-1，那么就将该动作的时间长度设置为该动作的总时间
            sec = (time_tup[1] - time_tup[0])//1000 if seconds == -1 else seconds
            x, y = self.choose(user_idx, action, sec, seed=seed, start_time=start_time, action_tup=action_tup, lap=lap)
            result[action] = (x, y)

        if return_dict:
            return result
        else:
            xs, ys = [], []
            for action, (x, y) in result.items():
                xs += x
                ys += y
            return xs, ys

    def chooseAll(self, user_idx):
        action_seconds = {action: -1 for action in self.activity_tag.keys()}
        return self.choose_mul(user_idx, action_seconds, start_time=-1)


class RawDataOwner(DataOwner):
    def __init__(self, path: str, time_window=2500):
        super(RawDataOwner, self).__init__(path, time_window=time_window)
        self.epsilon4window = 500

    def choose_in_one_loop(self, user_idx, action, seconds, start_time, lap, time_tup):
        action_dict = self.userset[user_idx][action]

        span = self.time_window - int(lap * self.time_window)
        time_series = list(range(start_time, start_time + seconds * 1000, span))
        num = len(time_series)
        err_nums = []

        xs = [[[[] for kdx in range(len(self.devices))] for jdx in range(len(self.sensor_freq.keys()))] for idx in range(num)]  # (num, sensor, location, dim)
        for sensor_idx, sensor in enumerate(self.sensor_freq.keys()):
            for dev_idx, dev in enumerate(self.devices):
                data = action_dict[sensor][dev]
                for num_idx, s_time in enumerate(time_series):
                    if data is None:
                        xs[num_idx][sensor_idx][dev_idx] = np.zeros(shape=(self.num_in_sensors[sensor], self.sensor_dim[sensor]))
                    else:
                        st = ((s_time - time_tup[0]) % (time_tup[1] - time_tup[0])) + time_tup[0]
                        et = ((s_time + self.time_window + self.epsilon4window - time_tup[0]) % (
                                    time_tup[1] - time_tup[0])) + time_tup[0]
                        if st < et:
                            dev_x = data[(data[:, 0] >= st) & (data[:, 0] < et), 1:]
                        else:
                            dev_x = data[(data[:, 0] >= st) | (data[:, 0] < et), 1:]
                        if len(dev_x) > self.num_in_sensors[sensor]:
                            dev_x = dev_x[:self.num_in_sensors[sensor]]
                        elif len(dev_x) < self.num_in_sensors[sensor]:
                            warnings.warn('one window data not enough {}.'.format(self.num_in_sensors[sensor]))
                            err_nums.append(num_idx)
                            continue
                        xs[num_idx][sensor_idx][dev_idx] = dev_x

                for err_idx in range(len(err_nums) - 1, -1, -1):
                    num = num - 1
                    time_series.pop(err_nums[err_idx])
                    xs.pop(err_nums[err_idx])
                err_nums.clear()

        ys = [self.activity_tag[action]] * num
        return xs, ys
#FftDataOwner类继承了DataOwner类,该类用于将数据集中的数据进行傅里叶变换
class FftDataOwner(DataOwner):
    def __init__(self, path: str, freq_rate, time_window=2500, allow_rate=0.8, expand_last=True):
        super(FftDataOwner, self).__init__(path, time_window=time_window)
        #epsilon4window = 50表示时间窗口的误差
        self.epsilon4window = 50
        self.expand_last = expand_last
        #allow_min是一个字典，key是传感器，value是该传感器在一个时间窗口内的最小采样点数,freq_rate是一个浮点数，表示采样点数的比例
        self.allow_min = {sensor: value * allow_rate for sensor, value in self.num_in_sensors.items()}
        #如果expand_last为True，那么就将传感器的维度扩大一倍,否则就将传感器的采样点数扩大一倍
        if expand_last:
            self.sensor_dim = {sensor: dim*2 for sensor, dim in self.sensor_dim.items()}
            self.num_in_sensors = {sensor: int(value * freq_rate) for sensor, value in self.num_in_sensors.items()}
        else:
            self.num_in_sensors = {sensor: int(value * freq_rate)*2 for sensor, value in self.num_in_sensors.items()}

    #在某个用户的某个活动中，按照span大小，从start_time开始，选择一num个时间窗口内的数据进行采样
    def choose_in_one_loop(self, user_idx, action, seconds, start_time, lap, time_tup):
        action_dict = self.userset[user_idx][action]
        #span是一个时间窗口的长度，lap是一个时间窗口的重叠率
        span = self.time_window - int(lap * self.time_window)
        #time_series是一个时间的列表，以span为步长，从start_time开始，start_time+seconds*1000结束，seconds表示当前活动的时间长度
        time_series = list(range(start_time, start_time + seconds * 1000, span))
        #num是时间窗口的个数
        num = len(time_series)
        err_nums = []
        #xs是一个时间窗口的数据，ys是一个时间窗口的标签
        #xs的shape是
        xs = [[[[] for kdx in range(len(self.devices))] for jdx in range(len(self.sensor_freq.keys()))] for idx in range(num)]  # (num, sensor, location, dim)
        #按照传感器的顺序遍历
        for sensor_idx, sensor in enumerate(self.sensor_freq.keys()):
            #按照设备的顺序遍历
            for dev_idx, dev in enumerate(self.devices):
                data = action_dict[sensor][dev]
                for num_idx, s_time in enumerate(time_series):
                    #将任意的 s_time 映射到了时间区间 time_tup 内。如果 s_time 超出了这个区间，那么它会被循环地映射回这个区间内
                    #st表示采样区间的开始时间，et表示采样区间的结束时间
                    st = ((s_time - time_tup[0]) % (time_tup[1] - time_tup[0])) + time_tup[0]
                    et = ((s_time + self.time_window + self.epsilon4window - time_tup[0]) % (time_tup[1] - time_tup[0])) + time_tup[0]
                    if st < et:
                        dev_x = data[(data[:, 0] >= st) & (data[:, 0] < et), 1:]
                    else:
                        dev_x = data[(data[:, 0] >= st) | (data[:, 0] < et), 1:]

                    if len(dev_x) < self.allow_min[sensor]:
                        warnings.warn('one window data not enough {}.'.format(self.allow_min[sensor]))
                        err_nums.append(num_idx)
                        continue
                    else:
                        #傅里叶变换后拼接在一起
                        dev_x = np.fft.fft(dev_x, axis=0, n=max(len(dev), self.num_in_sensors[sensor]))
                        if self.expand_last:
                            dev_x = dev_x[:self.num_in_sensors[sensor],:]
                            #将实部和虚部拼接在一起
                            dev_x = np.concatenate((dev_x.real, dev_x.imag), axis=-1)
                        else:
                            dev_x = dev_x[:self.num_in_sensors[sensor]//2, :]
                            dev_x = np.concatenate((dev_x.real, dev_x.imag), axis=-2)
                        
                        xs[num_idx][sensor_idx][dev_idx] = dev_x

                for err_idx in range(len(err_nums) - 1, -1, -1):
                    num = num - 1
                    time_series.pop(err_nums[err_idx])
                    xs.pop(err_nums[err_idx])
                err_nums.clear()

        ys = [self.activity_tag[action]] * num
        return xs, ys


class DeepSenseFftDataOwner(FftDataOwner):
    def __init__(self, path: str, time_window, T, allow_rate=0.5):
        super(DeepSenseFftDataOwner, self).__init__(path, time_window=time_window, freq_rate=1., allow_rate=allow_rate,
                                                    expand_last=False)
        self.epsilon4window = 0
        self.T = T

    def choose_in_one_loop(self, user_idx, action, seconds, start_time, lap, time_tup):
        action_dict = self.userset[user_idx][action]

        span = self.time_window*self.T - int(lap * self.time_window*self.T)
        time_series = list(range(start_time, start_time + seconds * 1000, span))

        xs = []  # (num, sensor, location, dim)
        for num_idx, s_time in enumerate(time_series):
            x_sensor = []
            is_break = False
            for sensor_idx, sensor in enumerate(self.sensor_freq.keys()):
                for dev_idx, dev in enumerate(self.devices):
                    data = action_dict[sensor][dev]
                    x_ts = []
                    for ti in range(self.T):
                        st = ((s_time+ self.time_window*ti - time_tup[0]) % (time_tup[1] - time_tup[0])) + time_tup[0]
                        et = ((s_time + self.time_window*(ti+1) + self.epsilon4window - time_tup[0]) % (
                                    time_tup[1] - time_tup[0])) + time_tup[0]
                        if st < et:
                            dev_x = data[(data[:, 0] >= st) & (data[:, 0] < et), 1:]
                        else:
                            dev_x = data[(data[:, 0] >= st) | (data[:, 0] < et), 1:]

                        if len(dev_x) < self.allow_min[sensor]:
                            warnings.warn('one window data not enough {}.'.format(self.allow_min[sensor]))
                            is_break = True
                            break
                        else:
                            dev_x = np.fft.fft(dev_x, axis=0, n=max(len(dev), self.num_in_sensors[sensor]))
                            dev_x = dev_x[:self.num_in_sensors[sensor] // 2, :]
                            dev_x = np.concatenate((dev_x.real, dev_x.imag), axis=-2)
                            x_ts.append(dev_x)
                    if is_break:
                        break
                    else:
                        x_sensor.append(np.array(x_ts))
                if is_break:
                    break

            if not is_break:
                xs.append(x_sensor)

        ys = [self.activity_tag[action]] * len(xs)
        return xs, ys

    def seconds_to_samples(self, seconds):
        return int((seconds * 1000) / (self.time_window * self.T))


def standard_array(data: np.ndarray) -> np.ndarray:
    ss = StandardScaler()
    dim = data.shape
    data = ss.fit_transform(data.reshape(-1, dim[-1])).reshape(dim)
    return data

def standard(xs :List[np.ndarray]):
    view_num = len(xs[0])
    xs_view = [standard_array(np.array([x[idx] for x in xs])) for idx in range(view_num)]
    xs = []
    for idx in range(len(xs_view[0])):
        x = []
        for view in range(view_num):
            x.append(xs_view[view][idx])
        xs.append(x)

    return xs
def split(xs: torch.Tensor, ys: torch.Tensor, vali_rate, test_rate, seed = None):
    indexes = np.array(range(len(ys)))
    split_dataset = {}
    if seed is not None:
        np.random.seed(seed)
    np.random.shuffle(indexes)

    vali = int(np.floor(vali_rate * len(ys)))
    test = int(np.floor(test_rate * len(ys)))

    if vali == 0:
        split_dataset['vali'] = None
    else:
        split_dataset['vali'] = data_util.ATTDataset(xs=[xs[idx] for idx in indexes[:vali]], ys=[ys[idx] for idx in indexes[:vali]])

    if test == 0:
        split_dataset['test'] = None
    else:
        split_dataset['test'] = data_util.ATTDataset(xs=[xs[idx] for idx in indexes[vali: vali + test]], ys=[ys[idx] for idx in indexes[vali: vali + test]])

    split_dataset['train'] = data_util.ATTDataset(xs=[xs[idx] for idx in indexes[vali + test:]], ys=[ys[idx] for idx in indexes[vali + test:]])

    return split_dataset


def to_tensor(data):
    if isinstance(data, (list, tuple)):
        return [to_tensor(x) for x in data]
    return torch.tensor(data).float()