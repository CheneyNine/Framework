import os
import argparse
import numpy as np
import pandas as pd

class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def generate_data_and_idx(data, x_offsets, y_offsets,
                          add_time_of_day=True,
                          add_day_of_week=True):
    """
    data: np.ndarray, shape (num_samples, num_nodes, feature_dim)
    timestamps: pd.DatetimeIndex, len == num_samples
    """
    # 1. 基本形状
    data = np.asarray(data)
    num_samples, num_nodes, feature_dim = data.shape
    # 设定起始时间和频率
    start_time = '2023-01-01 00:00:00'
    freq = '5T'  # "T" = minute，'5T' 表示每5分钟一个点
    periods = 365 * 24 * 60 // 5  # 一年 365 天，每小时 60 分钟，每 5 分钟一个点
    # 构造时间索引
    datetime_index = pd.date_range(start=start_time, periods=periods, freq=freq)
    print(datetime_index[:5])
    print(f"总时间步数: {len(datetime_index)}")

    # 示例：将第一个节点的数据变成 DataFrame（可扩展成多节点）
    df = pd.DataFrame(data[:, 0, :], index=datetime_index, columns=[f"feat_{i}" for i in range(4)])
    print(df.head())

    feature_list = [data]
    if add_time_of_day:
        time_ind = (df.index.values - df.index.values.astype('datetime64[D]')) / np.timedelta64(1, 'D')
        time_of_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
        feature_list.append(time_of_day)
    if add_day_of_week:
        dow = df.index.dayofweek
        dow_tiled = np.tile(dow, [1, num_nodes, 1]).transpose((2, 1, 0))
        day_of_week = dow_tiled / 7
        feature_list.append(day_of_week)

    data = np.concatenate(feature_list, axis=-1)
    print('data shape after feature concat:', data.shape)

    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive
    print('idx min & max:', min_t, max_t)
    idx = np.arange(min_t, max_t, 1)
    return data, idx



def generate_train_val_test(args):
    years = args.years.split('_')
    data=np.load('/root/code/Framework/dataset/sacramento_data.npy')
    print('original data shape:', data.shape) # (105120, 517, 4)
    # 筛选前面8928个时间点
    # data = data[:8928,:,:]
    print('data shape after year filter:', data.shape)

    seq_length_x, seq_length_y = args.seq_length_x, args.seq_length_y
    x_offsets = np.arange(-(seq_length_x - 1), 1, 1)
    y_offsets = np.arange(1, (seq_length_y + 1), 1)

    data, idx = generate_data_and_idx(data, x_offsets, y_offsets, args.tod, args.dow)
    print('final data shape:', data.shape, 'idx shape:', idx.shape)
    out_dir = args.dataset + '/' + args.years

    # np.save(os.path.join(out_dir, 'data_no_normalization'), data)
    num_samples = len(idx)
    num_train = round(num_samples * 0.6)
    num_val = round(num_samples * 0.2)   

    # split idx
    idx_train = idx[:num_train]
    idx_val = idx[num_train: num_train + num_val]
    idx_test = idx[num_train + num_val:]

    # normalize
    x_train = data[:idx_val[0] - args.seq_length_x, :, 0] 
    scaler = StandardScaler(mean=x_train.mean(), std=x_train.std())
    data[..., 0] = scaler.transform(data[..., 0])

    # save
    out_dir = '/root/code/Framework/dataset'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    np.savez_compressed(os.path.join(out_dir, 'his.npz'), data=data, mean=scaler.mean, std=scaler.std)

    np.save(os.path.join(out_dir, 'idx_train'), idx_train)
    np.save(os.path.join(out_dir, 'idx_val'), idx_val)
    np.save(os.path.join(out_dir, 'idx_test'), idx_test)
    print('data and idx saved to', out_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ca', help='dataset name')
    parser.add_argument('--years', type=str, default='2019', help='if use data from multiple years, please use underline to separate them, e.g., 2018_2019')
    parser.add_argument('--seq_length_x', type=int, default=12, help='sequence Length')
    parser.add_argument('--seq_length_y', type=int, default=12, help='sequence Length')
    parser.add_argument('--tod', type=int, default=1, help='time of day')
    parser.add_argument('--dow', type=int, default=1, help='day of week')
    
    args = parser.parse_args()
    generate_train_val_test(args)
