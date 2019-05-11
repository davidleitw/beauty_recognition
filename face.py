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

    def get_face_encoding(self, Img):
        return face_recognition.face_encodings(Img)

    def recognition(self, Img=None):
        #print(self.dataset.get_root_child())
        #Img_encoding = self.get_face_encoding(Img)
        avgdistance = []
        for idx, value in self.dataset.get_root_child():
            Imgs = os.listdir(os.path.join(data_path, value))
            Imgs.sort()
            ac = 0
            for i in range(10):
                compare_img = face_recognition.load_image_file(os.path.join(data_path, value, Imgs[i]))
                compare_img_encoding = self.get_face_encoding(compare_img)
                print(compare_img_encoding)
                #distance = face_recognition.face_distance()   
            

        






if __name__ == '__main__':
    Face_recognition = Face(known_data_path = data_path)
    #Face_recognition.recognition()
    Img = face_recognition.load_image_file(os.path.join(data_path, 'moe_five', '0024.jpg'))
    print(type(Img))
    #Img_encoding = Face_recognition.get_face_encoding(Img)
    Img_encoding = face_recognition.face_encodings(Img)
    print(Img_encoding)

