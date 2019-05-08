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
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.mx1 = nn.MaxPool2d((2, 2), stride=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.mx2 = nn.MaxPool2d((2, 2), stride=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.mx3 = nn.MaxPool2d((2, 2), stride=1)
    
        # print(self.features)
    def forward(self, Img):
        pass
    def testdata(self, Img):
        # Img = Img.view(3, 1350, 1080)
        # Img = torch.unsqueeze(Img, 1)
        print('In the test function')
        print(type(Img), Img.shape)
        print('After conv: ',self.conv1)
        output = self.conv1(Img)
        output = self.relu1(output)
        print('After conv1, output shape = ', output.shape)
        output = self.mx1(output)
        print('After maxpooling1, output shape = ', output.shape)

        output = self.conv2(output)
        output = self.relu2(output)
        print('After conv2, output shape = ', output.shape)
        output = self.mx2(output)
        print('After maxpooling2, output shape = ', output.shape)
       
        output = self.conv3(output)
        output = self.relu3(output)
        print('After conv3, output shape = ', output.shape)
        output = self.mx3(output)
        print('After mx3, output shape = ', output.shape)
    
        return Img

if __name__ == '__main__':
    Net = net()
    Imgpath = r'/media/davidlei/Transcend/Beauty_recognition/beauty_recognition/testdata/real__yami/'
    Img_ls = os.listdir(Imgpath)

   # Img = torch.from_numpy(cv2.imread(os.path.join(Imgpath, Img_ls[2]))).float()
   # print(Img.shape)
   # Img = Img.view(3, 1350, 1080)
   # print(Img.shape)
   # Img = torch.unsqueeze(Img, 0)
   # Img = Img.expand(1, 3, 1350, 1080)
    # Img = torch.reshape(Img, [16, 3, 1350, 1080])
   # print(Img.shape)
    # newimg = torch.ones(16, 3, 1350, 1080)
    # Net.testdata(Img)
   # Net.testdata(Img)


    # output = Net(Imgtensor)
