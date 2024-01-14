from train import database, stochastic_near
from util import Path
import os
import numpy as np
import warnings

warnings.filterwarnings("ignore")

print('pid: ', os.getpid())

dataset = database.Uci.NAME
path = Path.online_uci.format('all_all')
eval_seed = 4
time_window = int(2.5 * 1000)
allow_rate = 0.8
freq_rate = 1.
sup_seconds = 600
sup_fresh = 200
show_idx = 20
log_idx = 50
sup_batch_size = 128
max_run = 1
euc_cluster_limit = 400
cos_cluster_limit = 0.8
man_cluster_limit = 59000




def stochastic_near_train(samples, epochs, verbose=True):
    name = 'stochastic_near'
    stochastic_near.UciStochasticNearTrainer(
        lap=0.,
        log_path=Path.feder.format(dataset=dataset,
                                   samples=samples,
                                   model=name, name='log'),
        save_path=Path.feder.format(dataset=dataset,
                                    samples=samples,
                                    model=name, name='{}'),
        epochs=epochs, max_run=max_run,
        sup_batch_size=sup_batch_size,
        eval_seed=eval_seed, sup_fresh=sup_fresh,
        sup_seconds=sup_seconds,
        sup_samples=samples,
        time_window=time_window, allow_rate=allow_rate,
        freq_rate=freq_rate, data_path=path,cluster_limit=cos_cluster_limit). \
        start(verbose=verbose, show_idx=10, log_idx=log_idx)

if __name__ == '__main__':
    epochs = 1000
    seeds = [201, 3, 23, 202, 202, 2002, 25, 5, 103]
    idx = 3
    samples = database.Uci.SAMPLES_1
    np.random.seed(seeds[idx])
    stochastic_near_train(samples=samples, epochs=epochs, verbose=True)