import pickle
import numpy as np
import pandas as pd
# path = r'\root\project\fos\result\feder\uci\100 samples\stochastic_near\70.pkl'
# path2 = r'\root\project\fos\result\feder\uci\100 samples\stochastic_near\log.txt'
# f = open(path, 'rb')
# data = pickle.load(f)
# f = open(path2, 'w')
#
# f.write(str(data))
# f.close()

# import torch
# print(torch.cuda.is_available())

a = [{'a':2,'b':4},
     {'a':4,'b':6}]
b = np.mean([it['a'] for it in a])

print(b)