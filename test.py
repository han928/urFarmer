import numpy as np
import pandas as pd
import os

# vegs = []
#
# with open('../veg_list', 'r') as f:
#     for line in f:
#         vegs.append(line.split(':')[0].strip())
#
# file_list = os.listdir('.')
#
# for veg in vegs:
#     i = 0
#     for f in file_list:
#         if f.startswith(veg):
#             if i == 0:
#                 out = np.load(f)
#                 i+=1
#             else:
#                 out = np.r_[out, np.load(f)]
#                 i+=1
#     np.savetxt(veg+'.csv', out, delimiter=',')

# df = pd.read_csv(fname, header=None)
# np.savetxt('foo.csv', np_array, delimiter=',')


vegs = os.listdir('.')

df_list = []

for veg in vegs:
    if not veg.endswith('.py'):
        df_tmp = pd.DataFrame(np.load(veg))
        df_tmp['target'] = veg.split('.')[0]
        df_list.append(df_tmp)
