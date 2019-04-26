import face_recognition
import numpy
import time
import cv2
import os
import test

if __name__ == '__main__':
    path = '/media/davidlei/Transcend/Beauty_recognition/beauty_recognition/testdata/moe_five/'
    dataset = os.listdir(path)
    accuracy = []
    avgtime = 0
    count = 0
    for idx, picture in enumerate(dataset):
        ImgName, ImgNumber = test.Read_Img_File(picture)
        Img = face_recognition.load_image_file(os.path.join(path, ImgName))
        # img_face_encoding = face_recognition.face_encodings(Img)
        success = 0
        for img in dataset:  # Compare each image in dataset
            unImgName, unImgNumber = test.Read_Img_File(img)
            unknown_img = face_recognition.load_image_file(os.path.join(path, unImgName))
            # unknown_face_encoding = face_recognition.face_encodings(unknown_img)[0]

            start = time.time()
            flag = test.CompareFace(Img, unknown_img)
            if flag == 1:
                success = success + 1
            end = time.time()
            avgtime = avgtime + (end - start)
            count = count + 1
        # 紀錄每輪比較之後辨識的成功機率
        accuracy.append(success)

    for idx, record in enumerate(accuracy):
        print('In turn {}, ac% = {}'.format(idx, (record/len(dataset))))
    print('avgtime = {}'.format(avgtime/900))
    # print('count = {}'.format(count))