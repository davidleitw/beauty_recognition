import os
import torchvision
import cv2
import numpy as np
traindata_path = r'/home/davidlei/temp/beauty_recognition/train/'

class traindata():
    def __init__(self, train_root=None):
        self.root_path = train_root
        self.root_child = os.listdir(self.root_path)
        self.Dataset = []
        self.Labelset = []

    def __getitem__(self, idx=None):
        return self.root_child[idx]
    
    def get_childnum(self):
        return len(self.root_child)
    
    def get_childname(self):
        return list(self.root_child)

    def get_rootpath(self):
        return self.root_path

    def show_data(self):
        assert self.root_path != None, "root path can't be none!"
        print('root path = {}'.format(self.root_path))
        print('Include {} classes train data: {}'.format(len(self.root_child), self.root_child))

    def set_rootpath(self, train_root):
        assert os.path.isdir(train_root)
        self.root_path = train_root
        self.root_child = os.listdir(self.root_path)
        self.Dataset = []

    def loading_data(self):
        assert self.root_path != None, "root path can't be none!"
        imgcount = 0 
        for Class in self.root_child:
            print(Class)
            ClassesName = os.path.join(self.root_path, Class)
            ClassesImg = os.listdir(ClassesName)
            data = []
            assert len(ClassesImg)!= 0, "{} don't have img over there".format(os.path.join(ClassesName))
            for idx, img in enumerate(ClassesImg):
                Img = cv2.imread(os.path.join(ClassesName, img))
                data.append(Img)
                imgcount = imgcount + 1
            #print(len(data))
            self.Dataset.append(data)
            #Dataset.append(data)
        print('loading data finish, total classes {}, total img number {}'.format(imgcount, len(self.root_child)))
    
    def loading_label(self):
        assert self.root_path != None, "root path can't be none"
        labelcount = 0
        for Class in self.root_child:
            label = []
            for idx, img in enumerate(os.listdir(os.path.join(self.root_path, Class))):
                label.append(Class)
                labelcount = labelcount + 1 
            self.Labelset.append(label)
        print('loading label finish, total label with img {}'.format(labelcount))

    def get_img(self, classes_name=None, idx=None):
        for findptr in self.root_child:
            if findptr == classes_name:
                child_path = os.path.join(self.root_path, findptr)
                table = os.listdir(os.path.join(child_path))
                assert len(table)!=0, "len of table is zero, mean it did't exist picture in this table"
                for i, img in enumerate(table):
                    if i+1 == idx:
                        return cv2.imread(os.path.join(child_path, img))  
    def get_dataset(self):
        return self.Dataset

if __name__ == '__main__':
    dataset = traindata(traindata_path)
    dataset.show_data()
    dataset.loading_data()
    dataset.loading_label()
    Img = dataset.get_img(classes_name="moe_five", idx=2)
    name = dataset.get_childname()
    print(type(name))
    print(name)



