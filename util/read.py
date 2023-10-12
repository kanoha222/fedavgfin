# import pickle
# path = r'\root\project\fos\result\feder\uci\100 samples\stochastic_near\7log.pkl'
# path2 = r'\root\project\fos\result\feder\uci\100 samples\stochastic_near\log.txt'
# f = open(path, 'rb')
# data = pickle.load(f)
# f = open(path2, 'w')
#
# f.write(str(data))
# f.close()

a = [{'a':1},{'b':2}]
# a.append([4,5])
a += [{'c':3}]
print(a)