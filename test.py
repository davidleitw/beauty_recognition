from PIL import Image
import numpy as np
import time
import face_recognition
import os

def Read_Img_File(Imgpath=None):
    Imgname = Imgpath
    ImgNumber = Imgpath.split('.')
    ImgNumber = int(ImgNumber[0])
    # name and idx(number)
    return Imgname, ImgNumber

if __name__ == '__main__':
    path = '/media/davidlei/Transcend/Beauty_recognition/testdata/moe_five/'
    dataset = os.listdir(r'/media/davidlei/Transcend/Beauty_recognition/testdata/moe_five/') # 307
    dataset.sort()
    face = []
    # print(len(path))
    fcount = 0
    totaltime = 0

    for count,idx in enumerate(dataset):
        if count >= 10:
            break

        ImgName, ImgNumber = Read_Img_File(idx)
        print('{} {}'.format(ImgName, ImgNumber))

        Img = face_recognition.load_image_file(os.path.join(path, ImgName))

        start = time.time()
        local = face_recognition.face_locations(Img, model='cnn')
        face.append(local)
        end = time.time()
        totaltime = totaltime + (end - start)

        print('local = {}, num = {}'.format(local, count))
        if len(local) == 0:
            fcount = fcount + 1



    print('fcount = {}'.format(fcount))
    print(face)

