import torch
from torch import nn
from torch.utils.data import Dataset

class NetT(nn.Module):
    def __init__(self, in_shape, out_shape):
        super(NetT, self).__init__()
        layers = []
        sizes = [in_shape, 64, 128, 256, 512, 1024, 1024, 512, 256, 128, 64, out_shape]
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2: 
                layers.append(nn.LeakyReLU(inplace=True))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x) 

class EvData(Dataset):
    def __init__(self, indata, outdata):
        self.indata = indata
        self.outdata = outdata

    def __len__(self):
        return len(self.indata)

    def __getitem__(self, index):
        return (torch.from_numpy(self.indata[index]), 
                torch.from_numpy(self.outdata[index]))