import numpy as np
import torch
from torch.utils.data import Dataset


class Datapreprocessor:
    def __init__(self, dataset, input_len, output_len, stride=1, train_set_ratio=.6, validate_set_ratio=.2,random=False):
        self.num_sensors = dataset.shape[1]
        self.input_len = input_len
        self.output_len = output_len
        self.total_len = self.input_len + self.output_len
        self.std = np.std(dataset, axis=0).reshape([1, -1])
        self.mean = np.mean(dataset, axis=0).reshape([1, -1])
        self.dataset = (dataset - self.mean) / self.std
        self.time_encoding = self.generate_time_encoding(dataset.shape[0])

        self.num_samples = len(np.arange(0, dataset.shape[0] - self.total_len + 1, stride))
        sample_indices = np.arange(0, dataset.shape[0] - self.total_len + 1, stride)
        sample_window = np.arange(0, self.total_len)
        sample_index_mask = np.repeat(sample_indices, self.total_len).reshape(self.num_samples, -1) + sample_window
        if random:
            random_mask=np.random.permutation(self.num_samples)
            sample_index_mask=sample_index_mask[random_mask]
        samples = self.dataset[sample_index_mask]
        encoding = self.time_encoding[sample_index_mask]
        input = samples[:, :self.input_len, :]
        ground_truth = samples[:, self.input_len:, :]
        self.num_train_samples = int(self.num_samples * train_set_ratio)
        self.num_validate_samples = int(self.num_samples * validate_set_ratio)
        self.num_test_samples = self.num_samples - self.num_train_samples - self.num_validate_samples

        # train
        self.train_input = input[:self.num_train_samples]
        self.train_ground_truth = ground_truth[:self.num_train_samples]
        self.train_encoding = encoding[:self.num_train_samples]
        # validate
        self.validate_input = input[self.num_train_samples:self.num_train_samples + self.num_validate_samples]
        self.validate_ground_truth = ground_truth[
                                     self.num_train_samples:self.num_train_samples + self.num_validate_samples]
        self.validate_encoding = encoding[self.num_train_samples:self.num_train_samples + self.num_validate_samples]
        # test
        self.test_input = input[self.num_train_samples + self.num_validate_samples:]
        self.test_ground_truth = ground_truth[self.num_train_samples + self.num_validate_samples:]
        self.test_encoding = encoding[self.num_train_samples + self.num_validate_samples:]

    def load_train_samples(self, encoding=False):
        if encoding:
            return torch.Tensor(self.train_input), torch.Tensor(self.train_ground_truth), torch.Tensor(
                self.train_encoding)
        else:
            return torch.Tensor(self.train_input), torch.Tensor(self.train_ground_truth)

    def load_validate_samples(self, encoding=False):
        if encoding:
            return torch.Tensor(self.validate_input), torch.Tensor(self.validate_ground_truth), torch.Tensor(
                self.validate_encoding)
        else:
            return torch.Tensor(self.validate_input), torch.Tensor(self.validate_ground_truth)

    def load_test_samples(self, encoding=False):
        if encoding:
            return torch.Tensor(self.test_input), torch.Tensor(self.test_ground_truth), torch.Tensor(self.test_encoding)
        else:
            return torch.Tensor(self.test_input), torch.Tensor(self.test_ground_truth)

    def generate_time_encoding(self, len, dimension=6):
        if dimension > 6:
            print('dimension is too high')
            exit(1)
        time_stamp = np.zeros((len, 6))
        sec_of_min = (np.arange(60, dtype=float) - 30) / 60
        min_of_hour = (np.arange(60, dtype=float) - 30) / 60
        hour_of_day = (np.arange(24, dtype=float) - 12) / 24
        day_of_week = (np.arange(7, dtype=float) - 3) / 7
        day_of_month = (np.arange(31, dtype=float) - 15) / 31
        day_of_year = (np.arange(365, dtype=float) - 182) / 365
        for i in range(len):
            time_stamp[i, 0] = sec_of_min[i % 60]
            time_stamp[i, 1] = min_of_hour[i % 60]
            time_stamp[i, 2] = hour_of_day[i % 24]
            time_stamp[i, 3] = day_of_week[i % 7]
            time_stamp[i, 4] = day_of_month[i % 31]
            time_stamp[i, 5] = day_of_year[i % 365]
        return time_stamp[:, :dimension]


class LinearsDataset(Dataset):
    def __init__(self, input, ground_truth):
        super(LinearsDataset, self).__init__()
        self.input = input
        self.ground_truth = ground_truth

    def __getitem__(self, index):
        return self.input[index], self.ground_truth[index]

    def __len__(self):
        return self.input.shape[0]


class InformerDataset(Dataset):
    def __init__(self, input, ground_truth, encoding):
        super(InformerDataset, self).__init__()
        self.input = input
        self.ground_truth = ground_truth
        self.input_len = input.shape[1]
        self.input_y_shape = ground_truth.shape[1:]
        self.input_encoding = encoding[:, :self.input_len, :]
        self.output_encoding = encoding[:, self.input_len:, :]

    def __getitem__(self, index):
        input_y = torch.zeros(self.input_y_shape)
        return self.input[index], self.input_encoding[index], input_y, self.output_encoding[index], self.ground_truth[
            index]

    def __len__(self):
        return self.input.shape[0]

if __name__ == '__main__':
    import pickle
    import os
    data_root = 'E:\\forecastdataset\\pkl'
    input_len=10
    output_len=20
    stride=10
    dataset = pickle.load(open(os.path.join(data_root, 'ETTh1.pkl'), 'rb'))
    data_preprocessor = Datapreprocessor(dataset, input_len, output_len, stride=stride,random=True)
