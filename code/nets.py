import torch
import torch.nn as nn
from math import sqrt

class NetS(nn.Module):
    def __init__(self):
        super(NetS, self).__init__()
        self.conv_layers = nn.ModuleList(
            [nn.Conv3d(in_channels=1 if i == 0 else 128, out_channels=128 if i < 7 else 1, 
                       kernel_size=3, stride=1, padding=1, bias=False) for i in range(8)]
        )
        self.relu = nn.ReLU(inplace=True)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.conv_layers:
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))

    def forward(self, x):
        out = x
        for conv in self.conv_layers:
            out = conv(self.relu(out))
        return out + x