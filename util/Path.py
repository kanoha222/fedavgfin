root = '/home/jovyan/input/yb_uci_dataset/'
out = '/home/jovyan/work'
# raw data root

raw_uci =  root + '{group}/Inertial Signals/{loc}_{sensor}_{axis}_{group}.txt'
raw_uci_label =  root + '{group}/y_{group}.txt'
raw_uci_subject =  root + '{group}/subject_{group}.txt'


online_uci =   '/headless/Desktop/{}.pkl'

# feder result
feder = out + '/result/feder/{dataset}/{model}/fedavg/{name}.pkl'

# personalize
personalize =  root + '/result/personalize/{dataset}/{samples}_samples/{user}/{name}.pkl'
temp_personalize_csv =  root + '/result/personalize/{dataset}/csv/{dataset}_{name}.csv'

# arg
arg = root + '/result/arg/{dataset}/{samples}_samples/{arg}/{name}.pkl'

# loss
loss = root + '/result/loss/{dataset}/{samples}_samples/{name}.pkl'
