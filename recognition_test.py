import face_recognition
import time
import cv2
import os
import test

def show(Img):
    cv2.imshow('Img', Img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return Img

if __name__ == '__main__':
    path = '/media/davidlei/Transcend/Beauty_recognition/beauty_recognition/train/moe_five/'
    dataset = os.listdir(path)
    print(dataset)

    known_img = cv2.imread(os.path.join(path, dataset[1]))
    unknown_img = cv2.imread(os.path.join(path, dataset[2]))
    print(type(known_img), type(unknown_img))

    known_img_encoding = face_recognition.face_encodings(known_img)[0]
        # print(known_img_encoding)
    unknown_img_encoding = face_recognition.face_encodings(unknown_img)[0]

    results = face_recognition.face_distance(known_img_encoding, unknown_img_encoding)

    print(results)






