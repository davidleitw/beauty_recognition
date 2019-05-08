import torch
import torch.nn as nn
import cv2
import os
import numpy as np
traindata_path = r'/media/davidlei/Transcend/Beauty_recognition/beauty_recognition/train/real__yami/'
# https://blog.csdn.net/hbu_pig/article/details/81454503

def Conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def ImgtoTensor(Img):
    # Change BGR image to [batch, in_channels, Imgsize, Imgsize]
    Img = Img.view(3, Img.shape[0], Img.shape[1])
    Img = torch.unsqueeze(Img, 0)
    return Img

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
    def forward(self, x):
        # Conv1
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.mx1(x)
        # Conv2
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.mx2(x)
        # Conv3
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.mx3(x)

        return x
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

    Img_ls = os.listdir(traindata_path)

    Img = torch.from_numpy(cv2.imread(os.path.join(traindata_path, Img_ls[2]))).float()
    Img = ImgtoTensor(Img)
    print(Img.shape)
    # Net.testdata(Img)
    output = Net(Img)
    print(output.shape)

    # output = Net(Imgtensor)
