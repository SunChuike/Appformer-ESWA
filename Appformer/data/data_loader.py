import os
import ast
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')
import torch.nn as nn

def time_features1(dates):
    dates = pd.DataFrame(dates, columns=['date'])
    dates['month'] = dates.date.apply(lambda row: row.month, 1)
    dates['day'] = dates.date.apply(lambda row: row.day, 1)
    dates['weekday'] = dates.date.apply(lambda row: row.weekday(), 1)
    dates['hour'] = dates.date.apply(lambda row: row.hour, 1)
    dates['minute'] = dates.date.apply(lambda row: row.minute, 1)
    date = dates.iloc[0:4, 1:6]
    return date.values

class Dataset_Tsinghua(Dataset):
    def __init__(self, root_path, size, flag='train',
                 features='S', data_path='App_usage_trace.txt',
                 target='app_seq', scale=False, inverse=False, timeenc=0, freq='s', cols=None):
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val', 'test_pri']
        type_map = {'train': 0, 'val': 1, 'test': 2, 'test_pri': 3}
        self.set_type = type_map[flag]
        self.filename = flag + '.txt'
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path, self.filename), sep='\t')

    def __getitem__(self, index):
        data_raw = self.df_raw.iloc[index]

        app_seq = ast.literal_eval(data_raw['app_seq'])
        t = ast.literal_eval(data_raw['time_seq'])
        user = int(data_raw['user'])
        location_vectors_seq = ast.literal_eval(data_raw['location_vectors_seq'])

        with open('data/Tsinghua_new/time_division/division/Top1.txt', 'a') as f:
            f.write('app_seq: ' + str(app_seq) + '\n')
            f.write('time_seq: ' + str(t) + '\n')
            f.write('user: ' + str(user) + '\n')
            f.write('location_vectors_seq: ' + str(location_vectors_seq) + '\n')

        user = ','.join([str(user)] * 4)
        user = user.split(',')
        user = [eval(item) for item in user]
        user = np.array(user).reshape(4, 1)

        app_seq = np.array(app_seq)
        app_seq = np.expand_dims(app_seq, axis=1)

        location_vectors = [eval(item) for item in location_vectors_seq]
        location_vectors = np.array(location_vectors)

        time = pd.to_datetime(t)
        time_seq = time_features1(time)

        return app_seq, time_seq, location_vectors, user

    def __len__(self):
        return len(self.df_raw)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
