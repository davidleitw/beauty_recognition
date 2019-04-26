from PIL import Image
import numpy as np
import time
import cv2
import face_recognition
import os
SaveImg_path = r'/media/davidlei/Transcend/Beauty_recognition/beauty_recognition/diamond_test'

def Read_Img_File(Imgpath=None):
    '''
    :param Imgpath: img path
    :return: ImgName -> xxx.jpg, ImgNumber(index) -> int(xxx)
    '''
    Imgname = Imgpath
    ImgNumber = Imgpath.split('.')
    ImgNumber = int(ImgNumber[0])
    # name and idx(number)
    return Imgname, ImgNumber

def CompareFace(knownimg=None, unknownimg=None):
    known_encoding = face_recognition.face_encodings(knownimg)
    unknown_encoding = face_recognition.face_encodings(unknownimg)
    # print(len(known_encoding), len(unknown_encoding))
    results = face_recognition.compare_faces(known_encoding, unknown_encoding)
    flag = 1 if (results[0] == True) else 0
    return flag

def Drawface(Img=None, args=(0, 0, 0, 0), ImgName=None):
    # print(len(args))
    lockingface(Img, ImgName, args, mode='diamond', save=False, save_path=SaveImg_path)

def lockingface(Img=None, ImgName=None, point=(0, 0, 0, 0), mode='point', save=False, save_path=None):
    '''
    :param Img:
    :param ImgName:
    :param pointlist:
    :param mode:
    :param save:
    :param save_path:
    :return:
    '''
    if len(point) != 0:
        top = point[0][0]; right = point[0][1]; bottom = point[0][2]; left = point[0][3]
        Pointlist = [(left, top), (right, top), (left, bottom), (right, bottom)]

        if mode == 'point':
            try:
                for point in Pointlist:
                    cv2.circle(Img, point, 1, color=(255, 255, 0), thickness=4)
                if save:
                    cv2.imwrite(os.path.join(save_path, ImgName), Img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            except:
                print('Error, maybe cv2 is not working, can not draw the point on the picture')
        elif mode == 'diamond':
            try:
                cv2.line(Img, (int((right + left)/2), top-60), (right+40, int((top + bottom)/2)), color=(255, 255, 0), thickness=5)
                cv2.line(Img, (int((right + left)/2), top-60), (left-40, int((top + bottom)/2)), color=(255, 255, 0), thickness=5)
                cv2.line(Img, (int((right + left)/2), bottom+60), (left-40, int((top + bottom)/2)), color=(255, 255, 0), thickness=5)
                cv2.line(Img, (int((right + left)/2), bottom+60), (right+40, int((top + bottom)/2)), color=(255, 255, 0), thickness=5)
                if save:
                    cv2.imwrite(os.path.join(save_path, ImgName), Img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            except:
                print('Error, maybe cv2 is not working, can not draw the diamond on the picture')
    else:
        print('len(point) == 1, that mean there is not face in our picture')

if __name__ == '__main__':
    # print(dlib.DLIB_USE_CUDA)
    # print(dlib.cuda.get_num_devices())
    path = '/media/davidlei/Transcend/Beauty_recognition/beauty_recognition/testdata/moe_five/'
    fp = open('Record.txt', 'a')
    dataset = os.listdir(path)
    print(os.getcwd())
    # print(len(path))
    fcount = 0
    totaltime = 0

    for count,idx in enumerate(dataset):
        ImgName, ImgNumber = Read_Img_File(idx)
        # Img = face_recognition.load_image_file(os.path.join(path, ImgName))
        print('{} {}'.format(ImgName, ImgNumber))
        Img = cv2.imread(os.path.join(path, ImgName))

        start = time.time()
        # Use cuda to do face recognition, but i am not install dlib-cuda yet.
        local = face_recognition.face_locations(Img) # => use cuda for face_recognition
        # A list of tuples of found face locations in css (top, right, bottom, left) order
        # local = face_recognition.face_locations(Img)
        end = time.time()

        Drawface(Img, local, ImgName)
        totaltime = totaltime + (end - start)

        print('local = {}'.format(local))

        if len(local) == 0:
            fcount = fcount + 1


    avgtime = totaltime / 10
    print('ac% = {}'.format((len(dataset) - fcount)/len(dataset)))
    print('average time for each picture = {}'.format(avgtime))
    fp.write(" using fave_recognition.face_locations(Img, model='cnn') ")
    fp.write('ac% = {}'.format((len(dataset) - fcount)/len(dataset)))
    fp.write('average time for each picture = {}'.format(avgtime))

