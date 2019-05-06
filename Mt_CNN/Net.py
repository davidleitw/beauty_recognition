import torch
import torch.nn as nn
import cv2
import os
import numpy as np
# https://blog.csdn.net/hbu_pig/article/details/81454503

def Conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=5, stride=1, padding=1)
        # print(self.features)
    def forward(self, Img):
        pass
    def testdata(self, Img):
        # Img = Img.view(3, 1350, 1080)
        # Img = torch.unsqueeze(Img, 1)
        print('In the test function')
        print(type(Img), Img.shape)
        output = self.conv1(Img)
        print(output.shape)
        return Img

if __name__ == '__main__':
    Net = net()
    Imgpath = r'/media/davidlei/Transcend/Beauty_recognition/beauty_recognition/testdata/real__yami/'
    Img_ls = os.listdir(Imgpath)

    Img = torch.from_numpy(cv2.imread(os.path.join(Imgpath, Img_ls[2]))).float()
    print(Img.shape)
    Img = Img.view(3, 1350, 1080)
    print(Img.shape)
    Img = torch.unsqueeze(Img, 0)
    Img = Img.expand(16, 3, 1350, 1080)
    # Img = torch.reshape(Img, [16, 3, 1350, 1080])
    print(Img.shape)
    # newimg = torch.ones(16, 3, 1350, 1080)
    # Net.testdata(Img)
    Net.testdata(Img)

    # output = Net(Imgtensor)