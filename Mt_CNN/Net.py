import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class P_Net(nn.Module):
    def __init__(self):
        super(P_Net, self).__init__()
        self.features = nn.Sequential(
            OrderedDict([('conv1', nn.Conv2d(3, 10, 3, 1)),
                         ('prelu1', nn.PReLU(10)),
                         ('pool1', nn.MaxPool2d(2, 2, ceil_mode=True)),

                         ('conv2', nn.Conv2d(10, 16, 3, 1)),
                         ('prelu2', nn.PReLU(16)),

                         ('conv3', nn.Conv2d(16, 32, 3, 1)),
                         ('prelu3', nn.PReLU(32))
                         ])
            )
        self.conv4 = nn.Conv2d(32, 2, 1, 1)
        self.conv5 = nn.Conv2d(32, 4, 1, 1)

    def forward(self, x):
        x = self.features(x)
        x = self.conv4(x)
        x = self.conv5(x)
        results = F.softmax(x)
        return x, results

