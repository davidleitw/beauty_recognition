import os
import cv2
import numpy as np
import face_recognition
from dataset import traindata
data_path = r'../../temp/beauty_recognition/train/'

class Face(object):
    
    def __init__(self, known_data_path=None):
        assert known_data_path is not None and os.path.isdir(known_data_path), 'no data in path, or path have some error'    
        self.data_path = known_data_path
        self.dataset = traindata(known_data_path)
        self.dataset.loading_data()
        self.dataset.loading_label()
        self.dataset.show_data()

    def recognition(self, Img=None):
        for idx, child in enumerate(self.dataset.get_childname()):
            Imglist = os.listdir(os.path.join(data_path, child))
            Imglist.sort()
            
                
        






if __name__ == '__main__':
    Face_recognition = Face(known_data_path = data_path)
    Face_recognition.recognition()
