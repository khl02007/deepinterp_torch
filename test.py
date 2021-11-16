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
                 skip_adjacent: bool=True,start_sample: int=0, end_sample: int= 900000,
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
        self.num_samples_before = num_samples_before
        self.num_samples_after = num_samples_after
        self.skip_adjacent = skip_adjacent
        self.start_sample = start_sample
        self.end_sample = end_sample
        self.num_samples = end_sample-start_sample

        # load data 
        data = np.memmap(self.file_path, dtype=self.dtype)
        self.num_total_samples = int(data.size / self.num_chan)
        self.data = np.memmap(self.file_path, dtype=self.dtype,
                              shape=(self.num_total_samples, self.num_chan))
        
        if self.num_total_samples < self.end_sample:
            self.end_sample = self.num_total_samples
        
        # number of samples of a single example
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
        # if idx is greater than len does it automatically throw error?
        X = self.data[self.start_sample+idx:self.num_samples_single+idx, :].astype('float')
        if self.skip_adjacent:
            y = X[self.num_samples_before+1, :]
            X = np.delete(X, np.s_[self.num_samples_before:self.num_samples_before+3], axis=0)
        else:
            y = X[self.num_samples_before, :]
            X = np.delete(X, self.num_samples_before, axis=0)
        X = (X-self.mean)/self.std
        return torch.from_numpy(X), torch.from_numpy(y)


# loading data
# -----------------
# from torch.utils.data import DataLoader
# 
# training_data = EphysDataset(file_path='data', num_chan=128, 
#                              sampling_frequency=30e3, start_sample=0, end_sample=3*30e3)
# test_data = EphysDataset(file_path='data', num_chan=128, 
#                          sampling_frequency=30e3, start_sample=4*30e3, end_sample=5*30e3)
# train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
# test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
# train_features, train_labels = next(iter(train_dataloader))

from torch import nn

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.network = nn.Sequential(
            nn.Linear(in_features=4*64, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(),
            nn.Linear(out_features=32, out_features=16),
            nn.ReLU(),
            nn.Linear(out_features=16, out_features=8),
            nn.ReLU(),
            nn.Linear(out_features=8, out_features=4),
        )

    def forward(self, x):
        # don't call forward directly
        x = self.flatten(x)
        target_sample_value = self.network(x)
        return target_sample_value

# initialize model
# --------------------
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model = NeuralNetwork().to(device)
# X = torch.rand(1, 28, 28, device=device)
# logits = model(X)

# set hyper params
# --------------
# learning_rate = 1e-3
# batch_size = 64
# epochs = 1

# # Initialize the loss function
# loss_fn = nn.CrossEntropyLoss()

# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# def train_loop(dataloader, model, loss_fn, optimizer):
#     size = len(dataloader.dataset)
#     for batch, (X, y) in enumerate(dataloader):
#         # Compute prediction and loss
#         pred = model(X)
#         loss = loss_fn(pred, y)

#         # Backpropagation
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         if batch % 100 == 0:
#             loss, current = loss.item(), batch * len(X)
#             print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


# def test_loop(dataloader, model, loss_fn):
#     size = len(dataloader.dataset)
#     num_batches = len(dataloader)
#     test_loss, correct = 0, 0

#     with torch.no_grad():
#         for X, y in dataloader:
#             pred = model(X)
#             test_loss += loss_fn(pred, y).item()
#             correct += (pred.argmax(1) == y).type(torch.float).sum().item()

#     test_loss /= num_batches
#     correct /= size
#     print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

# loss_fn = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# epochs = 10
# for t in range(epochs):
#     print(f"Epoch {t+1}\n-------------------------------")
#     train_loop(train_dataloader, model, loss_fn, optimizer)
#     test_loop(test_dataloader, model, loss_fn)
# print("Done!")

# torch.save(model, 'model.pth')
# model = torch.load('model.pth')
