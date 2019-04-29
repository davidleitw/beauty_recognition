import face_recognition
import numpy
import time
from PIL import Image
import cv2
import os
import test

if __name__ == '__main__':
    path = '/media/davidlei/Transcend/Beauty_recognition/beauty_recognition/testdata/moe_five/'
    

    dataset = os.listdir(path)
    Img = []
    Idx = []
    for num, idx in enumerate(dataset):
        img = face_recognition.load_image_file(os.path.join(path, idx))
        local = face_recognition.face_locations(img, model='cnn')
        Img.append(img)
        if len(local) != 0:
            Idx.append(num)

    # print(Img[0])
    print(Img[0].shape)
    print(Img[1].shape)
    
    try:
        img1 = face_recognition.face_encodings(Img[1])[0]
        img2 = face_recognition.face_encodings(Img[2])[0]
        unknown_img3 = face_recognition.face_encodings(Img[3])[0]
    except IndexError:
        print('check the file')

    known_faces = [img1, img2]

    results = face_recognition.compare_faces(known_faces, unknown_img3)

    # print('results[0] = {}'.format(results[0]))
    # print('results[1] = {}'.format(results[1]))





