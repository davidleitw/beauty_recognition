import os
import torchvision
import cv2
import numpy as np
traindata_path = r'/media/davidlei/Transcend/Beauty_recognition/beauty_recognition/train/'


class traindata():
    def __init__(self, train_root=None):
        self.root_path = train_root
        self.root_child = os.listdir(self.root_path)
        self.Dataset = []
        self.Labelset = []

    def __getitem__(self, idx):
        return self.root_child[idx]

    def show_data(self):
        assert self.root_path != None, "root path can't be none!"
        print('root path = {}'.format(self.root_path))
        print('Include {} classes train data:\n{}'.format(len(self.root_child), self.root_child))

    def set_rootpath(self, train_root):
        self.root_path = train_root
        self.root_child = os.listdir(self.root_path)
        self.Dataset = []

    def loading_data(self):
        assert self.root_path != None, "root path can't be none!"
       
        for Class in self.root_child:
            print(Class)
            ClassesName = os.path.join(self.root_path, Class)
            ClassesImg = os.listdir(ClassesName)
            data = []
            for idx, img in enumerate(ClassesImg):
                Img = cv2.imread(os.path.join(ClassesName, img))
                data.append(Img)
            #print(len(data))
            self.Dataset.append(data)
            #Dataset.append(data)
        print(len(self.Dataset[0]))
    
    def loading_label(self):
        assert self.root_path != None, "root path can't be none"

        for Class in self.root_child:
            label = []
            for idx, img in enumerate(os.listdir(os.path.join(self.root_path, Class))):
                label.append(Class)
            self.Labelset.append(label)


if __name__ == '__main__':
    dataset = traindata(traindata_path)
    dataset.show_data()
    dataset.loading_data()
    dataset.loading_label()
    print(dataset.Dataset)
    print(dataset.Labelset)




