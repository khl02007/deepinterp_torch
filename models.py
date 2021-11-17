from torch import nn

class FullyConnectedFlatten(nn.Module):
    def __init__(self, num_samples_before, num_samples_after, num_chan):
        super().__init__()
        self.flatten = nn.Flatten()
        self.network = nn.Sequential(
            nn.Linear(in_features=num_chan*(num_samples_before+num_samples_after), out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=8),
            nn.ReLU(),
            nn.Linear(in_features=8, out_features=4),
        )

    def forward(self, x):
        # don't call forward directly
        x = self.flatten(x)
        target_sample_value = self.network(x)
        return target_sample_value