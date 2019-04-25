from PIL import Image
import numpy as np
import time
import cv2
import face_recognition
import os

def Read_Img_File(Imgpath=None):
    Imgname = Imgpath
    ImgNumber = Imgpath.split('.')
    ImgNumber = int(ImgNumber[0])
    # name and idx(number)
    return Imgname, ImgNumber

def Drawface(Img, args):
    # print(len(args))
    if len(args) != 0:
        top = args[0][0]; right = args[0][1]; bottom = args[0][2]; left = args[0][3]
        print('top = {}, right = {}, bottom = {}, left = {}'.format(top, right, bottom, left))
        cv2.line(Img, (int((right + left)/2), top-60), (right+40, int((top + bottom)/2)), color=(255, 255, 0), thickness=5)
        cv2.line(Img, (int((right + left)/2), top-60), (left-40, int((top + bottom)/2)), color=(255, 255, 0), thickness=5)
        cv2.line(Img, (int((right + left)/2), bottom+60), (left-40, int((top + bottom)/2)), color=(255, 255, 0), thickness=5)
        cv2.line(Img, (int((right + left)/2), bottom+60), (right+40, int((top + bottom)/2)), color=(255, 255, 0), thickness=5)
        cv2.imshow('temp', Img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    # print(dlib.DLIB_USE_CUDA)
    # print(dlib.cuda.get_num_devices())
    path = '/media/davidlei/Transcend/Beauty_recognition/beauty_recognition/testdata/moe_five/'
    dataset = os.listdir(r'/media/davidlei/Transcend/Beauty_recognition/beauty_recognition/testdata/moe_five/')
    dataset.sort()
    face = []
    # print(len(path))
    fcount = 0
    totaltime = 0

    for count,idx in enumerate(dataset):

        ImgName, ImgNumber = Read_Img_File(idx)
        # Img = face_recognition.load_image_file(os.path.join(path, ImgName))
        Img = cv2.imread(os.path.join(path, ImgName))

        start = time.time()
        # Use cuda to do fa recognition, but i am not install dlib-cuda yet.
        # local = face_recognition.face_locations(Img, model='cnn') => use cuda for face_recognition
        # A list of tuples of found face locations in css (top, right, bottom, left) order
        local = face_recognition.face_locations(Img)
        Drawface(Img, local)
        end = time.time()
        totaltime = totaltime + (end - start)

        print('local = {}'.format(local))

        if len(local) == 0:
            fcount = fcount + 1


    avgtime = totaltime / 10
    print('ac% = {}'.format((len(dataset) - fcount)/len(dataset)))
    print('average time for each picture = {}'.format(avgtime))


