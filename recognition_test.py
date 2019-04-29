import face_recognition
import numpy
import time
from PIL import Image
import cv2
import os
import test

def show(Img):
    cv2.imshow('Img', Img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return Img

if __name__ == '__main__':
    path = '/media/davidlei/Transcend/Beauty_recognition/beauty_recognition/testdata/moe_five/'
    dataset = os.listdir(path)
    print(dataset)

    known_img = cv2.imread(os.path.join(path, dataset[1]))
    unknown_img = cv2.imread(os.path.join(path, dataset[2]))

    known_img = cv2.resize(known_img, (1920, 768))
    unknown_img = cv2.resize(unknown_img, (1920, 768))

    try:
        known_img_encoding = face_recognition.face_encodings(known_img)
        unknown_img_encoding = face_recognition.face_encodings(unknown_img)

    except:
        print('check the file')

    results = face_recognition.compare_faces(known_img_encoding, unknown_img_encoding)
    print(results)






