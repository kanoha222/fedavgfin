from abc import abstractmethod
from util import data_process


class Database:
    def __init__(self, sup_seconds, sup_fresh, sup_samples, time_window,
                 allow_rate, freq_rate, data_path):
        self.sup_seconds = sup_seconds
        self.sup_fresh = sup_fresh
        self.sup_samples = sup_samples
        self.time_window = time_window
        self.allow_rate = allow_rate
        self.freq_rate = freq_rate
        self.data_path = data_path
        self.owner = None

    @abstractmethod
    def make_owner(self):
        pass



class Uci(Database):
    NAME = 'uci'
    SECCONDS_PER_USER = 472
    SAMPLES_PER_USRE = 187
    CLIENTS = 30

    SAMPLES_1 = 100
    SAMPLES_2 = 140
    SAMPLES_3 = 180

    STAFF_1S = 2
    STAFF_2S = 3
    STAFF_1 = 4
    STAFF_2 = 6
    STAFF_3 = 8

    def __init__(self, **kwargs):
        super(Uci, self).__init__(**{key: value for key, value in kwargs.items() if key in Database.__init__.__code__.co_varnames})

    def make_owner(self):
        self.owner = data_process.FftDataOwner(path=self.data_path, freq_rate=self.freq_rate,
                                               time_window=self.time_window,
                                               allow_rate=self.allow_rate, expand_last=True)
