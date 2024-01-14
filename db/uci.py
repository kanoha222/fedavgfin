from util import utils, Path
import numpy as np

activity_tag = {'walking': 0, 'upstairs': 1, 'downstairs': 2, 'sitting': 3, 'standing': 4, 'laying': 5}
sensor_freq = {'acc': 50, 'gyr': 50}
sensor_dim = {'acc': 3, 'gyr': 3}
locations = ['waist']

stamp = 1

def get_subject(group) -> np.ndarray:
    subj = utils.read(Path.raw_uci_subject.format(group=group))
    subj = [value.strip() for value in subj.split('\n') if value.strip() != '']
    subj = np.array(subj, dtype=int)

    return subj


def get_data(loc, group, sensor, axis) -> np.ndarray:
    data = utils.read(Path.raw_uci.format(group=group, sensor=sensor, axis=axis, loc=loc))
    data = [value.strip() for value in data.split('\n') if value.strip() != '']
    for idx in range(len(data)):
        data[idx] = [value.strip() for value in data[idx].split(' ') if value.strip() != '']
    data = np.array(data, dtype=float)

    return data


def get_label(group) -> np.ndarray:
    label = utils.read(Path.raw_uci_label.format(group=group))
    label = np.array([value.strip() for value in label.split('\n') if value.strip() != ''], dtype=int)

    return label - 1


def no_lap_index(data: np.ndarray):
    data_0 = data[:-1, 64:]
    data_1 = data[1:, :64]
    return np.nonzero((data_0 != data_1).any(axis=1))[0]
#该方法用于找到数据中的不同的位置，用于切分数据
def flatten(data :np.ndarray):
    xs = []
    xs += data[0].tolist()
    #此处循环用于判断是否有重复的数据，如果有重复的数据则不加入，否则加入
    for idx in range(1, len(data)):
        #
        if np.all(data[idx-1][64:] == data[idx][:64]): # lap
            xs += data[idx][64:].tolist()
        else: # no lap
            xs += data[idx].tolist()

    return xs

#该方法用于获取时间戳
def get_time(length, freq):
    result = []
    global stamp
    for idx in range(length):
        result.append(stamp)
        stamp = stamp + int(1000 // freq)

    return result

def extract_sensor(sensor_dict, index):
    sensor_csv_dict = {sensor: {} for sensor in sensor_freq.keys()}
    timeline = []
    for sensor, values in sensor_dict.items():
        data = []
        #values表示传感器的数据（三维），index表示当前用户的索引
        for value in values:
            #flatten方法去除重复的数据
            data.append(flatten(value[index]))

        if len(timeline) == 0:  # initiate
            timeline = get_time(len(data[0]), sensor_freq[sensor])
        data.insert(0, timeline)
        #此处用于将数据轴转换，第一列是时间，后面是xyz轴的数据
        data = np.array(data).swapaxes(0, 1)
        sensor_csv_dict[sensor][locations[0]] = data

    return sensor_csv_dict, (min(timeline), max(timeline))
    #用于计算每个用户的数据长度，以及每个用户的数据起始时间
def extract_action(label, start, sensor_dict):
    action_dict = {}
    num_dict = {}
    for y in np.unique(label):
        #获取当前用户某个动作的索引，start表示当前用户的数据起始索引
        index = np.nonzero(label == y)[0] + start

        action = [key for key, value in activity_tag.items() if value == y][0]
        action_dict[action], num_dict[action] = extract_sensor(sensor_dict, index)

    return action_dict, num_dict


def extract(subject, label, sensor_dict):
    userset = []
    userlen = []
    #对每个用户进行数据提取
    for user in np.unique(subject):
        index = np.nonzero(subject == user)[0]
        start = min(index)
        end = max(index) + 1
        if (end - start) != len(index):
            raise Exception()
        #action_dict是一个字典，key是动作，value是一个字典，key是传感器，value是一个字典，key是位置，value是一个二维数组，第一列是时间，后面是数据
        action_dict, num_dict = extract_action(label[index], start, sensor_dict)
        # sensor_dict: {传感器：[[x][y][z]}
        # action_dict: {行为: {sensor_dict}}
        # userset: [action_dict_user1,, , action_dict_userN]
        #num_dict_user1：某个用户的每个行为的时间跨度
        #userlen: [num_dict_user1, , , num_dict_userN],每个用户的每个行为的时间跨度
        userset.append(action_dict)
        userlen.append(num_dict)

    return userset, userlen


def save(name):
    userset = []
    userlen = []
    for group in ['train', 'test']:
        subject = get_subject(group)
        label = get_label(group)
        sensor_dict = {}
        #sensor_dict求的是('acc', 'total')和('gyr', 'body')下xyz轴数据，没有（acc，body）
        for sensor, loc in [('acc', 'total'), ('gyr', 'body')]:
            sensor_dict[sensor] = []
            for axis in ['x', 'y', 'z']:
                sensor_dict[sensor].append(get_data(loc, group, sensor, axis))

        uset, ulen = extract(subject, label, sensor_dict)
        userset += uset
        userlen += ulen

    utils.write_pkl(Path.online_uci.format(name), [userset, userlen, activity_tag, sensor_freq, sensor_dim, locations])

save('all_all')
