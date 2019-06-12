import cv2
import torch
import torchvision
import numpy as np
import torch.nn as nn
from dataset import traindata
from Net import beauty_net

train_data_path = r"/media/davidlei/Transcend/Beauty_recognition/beauty_recognition/train/"

if __name__ == '__main__':
    
    dataloader = traindata(train_data_path)
    dataloader.loading_data()
    dataloader.loading_label()
    dataloader.show_data()
    
    print(dataloader.root_path)
    print(dataloader.Labelset)
    print(dataloader.Dataset)
    print(len(dataloader.Dataset[0]))
    #print(dataloader.get_da`taset)
    


    train_loader = torch.utils.data.DataLoader(dataloader.Dataset) 
    print(train_loader)
    print(len(train_loader))
    











