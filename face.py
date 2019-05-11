import os
import cv2
import numpy as np
import face_recognition
from dataset import traindata
data_path = r''

class Face(object):
    def __init__(self, known_data_path=None):
    	assert known_data_path is not None and os.path.isdir(known_data_path), 'no data in path, or path have some error'
        self.data_path = known_data_path
        self.dataset = traindata(known_data_path)
        self.dataset.loading_data()
        self.dataset.loading_label()
        
    def recognition(self, Img=None):
        pass






if __name__ == '__main__':
    
