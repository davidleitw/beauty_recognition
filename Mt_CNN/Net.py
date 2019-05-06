import torch
import torch.nn as nn
import cv2
import numpy as np

class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()
        self.features = nn.Sequential(
            # Conv1
            nn.Conv2d(in_channels=3, out_channels=16,
                      kernel_size=5, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # Conv2
            nn.Conv2d(in_channels=16, out_channels=32,
                      kernel_size=4, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # Conv3
            nn.Conv2d(in_channels=32, out_channels=32,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        print(self.features)
    def forward(self, Img):
        x = self.features(Img)
        return x

if __name__ == '__main__':
    Net = net()
    Imgpath = r''
