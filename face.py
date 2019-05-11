import face_recognition
from dataset import traindata
import numpy as np
import cv2 

class Face(object):
    def __init__(self, known_data_path=None):
    	assert known_data_path is not None, 'we need more data!!!'
        self.data_path = known_data_path
        self.dataset = traindata(known_data_path)
    
    def recognition(self, Img=None):
        
