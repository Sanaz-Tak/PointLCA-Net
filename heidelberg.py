import numpy as np
from torch.utils.data import Dataset
import tables
import torch

#fileh = tables.open_file('/Users/sxm788/PycharmProjects/Hiedelberg/dataset/shd_train.h5', mode='r')
#units = fileh.root.spikes.units
#times = fileh.root.spikes.times
#labels = fileh.root.labels


class HeidelbergDataset(Dataset):
    def __init__(
        self, path='dataset',
        train=True,
        transform=None,
        target_transform=None
    ):
        super(HeidelbergDataset, self).__init__()
        if train:
            fileh = tables.open_file(path, mode='r')
        else:
            fileh = tables.open_file(path, mode='r')
        self.units = fileh.root.spikes.units
        self.times = fileh.root.spikes.times
        self.labels = fileh.root.labels

    def __getitem__(self, i):
        label = int(self.labels[i])
        '''
        coo = [[] for i in range(2)]
        coo[0].extend(self.times[i].astype(int))
        coo[1].extend(self.units[i].astype(int))
        return coo, label
        '''
        x_event = self.units[i].astype(int)
        y_event = x_event
        #y_event = np.ones_like(x_event) * 0.5
        t_event = np.round(1000 * self.times[i]).astype(int)

        x = torch.tensor(x_event, dtype=torch.float32)
        x = x/torch.max(x)
        y = torch.tensor(y_event, dtype=torch.float32)
        y = y / torch.max(y)
        t = torch.tensor(t_event, dtype=torch.float32)
        t = t / torch.max(t)
        ev = torch.cat((x.unsqueeze(0), y.unsqueeze(0), t.unsqueeze(0)), dim=0).squeeze()
        
        data_size = ev.size(1)
        if data_size >= 4096*4:
            window_size = data_size // 32
            windows = [ev[:, i * window_size: (i + 1) * window_size] for i in range(32)]
            nn = 32
        elif data_size >= 4096*2:
            window_size = data_size // 16
            windows = [ev[:, i * window_size: (i + 1) * window_size] for i in range(16)]
            nn = 64
        elif data_size >= 4096:
            window_size = data_size // 8
            windows = [ev[:, i * window_size: (i + 1) * window_size] for i in range(8)]
            nn = 128
        elif data_size >= 2048:
            window_size = data_size // 4
            windows = [ev[:, i * window_size: (i + 1) * window_size] for i in range(4)]
            nn = 256
        elif data_size >= 1024:
            window_size = data_size // 2
            windows = [ev[:, i * window_size: (i + 1) * window_size] for i in range(2)]
            nn = 512
        else:
            windows = [torch.cat((ev, ev[:, -1:].repeat(1, 1024 - ev.shape[1])), dim=1)]
            nn = 1024

        sampled_parts = [window[:,torch.randperm(nn)] for window in windows]

        ev_all = torch.cat(sampled_parts, dim=1)
        idx = np.arange(ev_all.shape[1])
        np.random.shuffle(idx)
        idx = idx[0:1024]
        return ev_all[:,idx], label



        #if ev.shape[1] <= 14346:
            #ev = torch.cat((ev, ev[:, -1:].repeat(1, 14346 - ev.shape[1])), dim=1)
        #idx = np.arange(ev.shape[1])
        #np.random.shuffle(idx)
        #idx = idx[0:14346]
        #return ev[:, idx], label

        # # empty_tensor = torch.zeros(700, t_event.max())
        # empty_tensor = torch.zeros(700, 1369)  #max is 1369 with avg of 715.6
        # valid_ind = np.argwhere(
        #     (x_event < empty_tensor.shape[0])
        #     & (t_event < empty_tensor.shape[1])
        #     & (x_event >= 0)
        #     & (t_event >= 0)
        # )
        # empty_tensor[x_event[valid_ind], t_event[valid_ind]] = 1
        #
        # return empty_tensor, label

    def __len__(self):
        return len(self.labels)

    def fill_tensor(
        self, empty_tensor,
        sampling_time=1, random_shift=False, binning_mode='OR'
    ):  # Sampling time in ms
        """Returns a numpy tensor that contains the spike events sampled in
        bins of ``sampling_time``. The tensor is of dimension
        (channels, height, width, time) or``CHWT``.

        Parameters
        ----------
        empty_tensor : numpy or torch tensor
            an empty tensor to hold spike data .
        sampling_time : float
            the width of time bin. Defaults to 1.
        random_shift : bool
            flag to randomly shift the sample in time. Defaults to False.
        binning_mode : str
            the way spikes are binned. Options are 'OR'|'SUM'. If the event is
            graded binning mode is overwritten to 'SUM'. Defaults to 'OR'.

        Returns
        -------
        numpy or torch tensor
            spike tensor.

        Examples
        --------

        >>> spike = td_event.fill_tensor( torch.zeros((2, 240, 180, 5000)) )
        """
        if random_shift is True:
            t_start = np.random.randint(
                max(
                    int(self.t.min() / sampling_time),
                    int(self.t.max() / sampling_time) - empty_tensor.shape[3],
                    empty_tensor.shape[3] - int(self.t.max() / sampling_time),
                    1,
                )
            )
        else:
            t_start = 0

        x_event = np.round(self.x).astype(int)
        c_event = np.round(self.c).astype(int)
        t_event = np.round(self.t / sampling_time).astype(int) - t_start
        if self.graded:
            payload = self.p
            binning_mode = 'SUM'

        # print('shifted sequence by', t_start)

        if self.dim == 1:
            valid_ind = np.argwhere(
                (x_event < empty_tensor.shape[2])
                & (c_event < empty_tensor.shape[0])
                & (t_event < empty_tensor.shape[3])
                & (x_event >= 0)
                & (c_event >= 0)
                & (t_event >= 0)
            )
            if binning_mode.upper() == 'OR':
                empty_tensor[
                    c_event[valid_ind],
                    0,
                    x_event[valid_ind],
                    t_event[valid_ind]
                ] = payload if self.graded is True else 1 / sampling_time
            elif binning_mode.upper() == 'SUM':
                empty_tensor[
                    c_event[valid_ind],
                    0,
                    x_event[valid_ind],
                    t_event[valid_ind]
                ] += payload if self.graded is True else 1 / sampling_time
            else:
                raise Exception(
                    f'Unsupported binning_mode. It was {binning_mode}'
                )

        elif self.dim == 2:
            y_event = np.round(self.y).astype(int)
            valid_ind = np.argwhere(
                (x_event < empty_tensor.shape[2])
                & (y_event < empty_tensor.shape[1])
                & (c_event < empty_tensor.shape[0])
                & (t_event < empty_tensor.shape[3])
                & (x_event >= 0)
                & (y_event >= 0)
                & (c_event >= 0)
                & (t_event >= 0)
            )

            if binning_mode.upper() == 'OR':
                empty_tensor[
                    c_event[valid_ind],
                    y_event[valid_ind],
                    x_event[valid_ind],
                    t_event[valid_ind]
                ] = payload if self.graded is True else 1 / sampling_time
            elif binning_mode.upper() == 'SUM':
                empty_tensor[
                    c_event[valid_ind],
                    y_event[valid_ind],
                    x_event[valid_ind],
                    t_event[valid_ind]
                ] += payload if self.graded is True else 1 / sampling_time
            else:
                raise Exception(
                    'Unsupported binning_mode. It was {binning_mode}'
                )

        return empty_tensor
