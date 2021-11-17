import torch
from torch.utils.data import Dataset
import numpy as np

class EphysDataset(Dataset):
    """Class for loading electrophysiology data
    Currently works only for data in binary format saved as (samples, chans)
    __getitem__ returns (example, target sample value) 
        example is the samples before and after target sample, (samples, chans)
        target sample value is (chans,)
    """
    def __init__(self, file_path: str, num_chan: int, sampling_frequency: float=30e3,
                 dtype='int16', num_samples_before: int=32, num_samples_after: int=32,
                 skip_adjacent: bool=True, start_sample: int=0, end_sample: int= 900000,
                 transform: bool=True):
        """Initialize EphysDataset object

        Parameters
        ----------
        file_path : str
            [description]
        num_chan : int
            [description]
        sampling_frequency : float, optional
            [description], by default 30e3
        dtype : str, optional
            [description], by default 'int16'
        num_samples_before : int, optional
            [description], by default 32
        num_samples_after : int, optional
            [description], by default 32
        skip_adjacent : bool, optional
            [description], by default True
        start_sample : int, optional
            [description], by default 0
        end_sample : int, optional
            [description], by default 900000
        """
        
        self.file_path = file_path
        self.num_chan = num_chan
        self.sampling_frequency = sampling_frequency
        self.dtype = dtype
        self.num_samples_before = int(num_samples_before)
        self.num_samples_after = int(num_samples_after)
        self.skip_adjacent = skip_adjacent
        self.start_sample = int(start_sample)
        self.end_sample = int(end_sample)
        self.num_samples = self.end_sample-self.start_sample

        # load data 
        data = np.memmap(self.file_path, dtype=self.dtype)
        self.num_total_samples = int(data.size / self.num_chan)
        self.data = np.memmap(self.file_path, dtype=self.dtype,
                              shape=(self.num_total_samples, self.num_chan))
        
        if self.num_total_samples < self.end_sample:
            self.end_sample = self.num_total_samples
        
        # number of samples of a chunk of data for a single (example, target) pair
        if self.skip_adjacent:
            self.num_samples_single = self.num_samples_before + self.num_samples_after + 3
        else:
            self.num_samples_single = self.num_samples_before + self.num_samples_after + 1
        
        if transform:
            chunk = self.data[0:int(np.min([10*self.sampling_frequency,
                                            self.num_total_samples]))]
            self.mean = np.mean(chunk, axis=0)
            self.std = np.std(chunk, axis=0)
        else:
            self.mean = 0
            self.std = 1
        
    def __len__(self):
        return self.num_samples - self.num_samples_single + 1

    def __getitem__(self, idx):
        # get data chunk
        if idx>len(self)-1:
            raise IndexError
        X = self.data[self.start_sample+idx:self.start_sample+idx+self.num_samples_single, :]
        # transfrom
        X = (X-self.mean)/self.std
        # make pytorch Tensor
        X = torch.from_numpy(X)
        if self.skip_adjacent:
            y = X[self.num_samples_before+1, :]
            X = np.append(X[:self.num_samples_before,:], X[self.num_samples_before+3:,:], axis=0)
        else:
            y = X[self.num_samples_before, :]
            X = np.append(X[:self.num_samples_before,:], X[self.num_samples_before+1:,:], axis=0)
        return X, y

