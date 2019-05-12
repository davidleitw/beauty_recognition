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

    def recognition(self, Img=None, compare_num=15):
        #print(self.dataset.get_root_child())
        Img_encoding = self.get_face_encoding(Img)
        avgdistance = []
        
        for idx, value in self.dataset.get_root_child():
            Imgs = os.listdir(os.path.join(data_path, value))
            Imgs.sort()
            Ac = np.zeros((1, compare_num))
            count = 0
            for i in range(15):
                compare_img = face_recognition.load_image_file(os.path.join(data_path, value, Imgs[i]))                
                if len(self.get_face_encoding(compare_img)) == 0:
                    continue
                else :
                    compare_img_encoding = self.get_face_encoding(compare_img)[0]
                Ac[0, count] = face_recognition.face_distance(compare_img_encoding, Img_encoding)
                print('Ac = {}'.format(Ac[0, count]))
                count = count + 1      
                if count >= compare_num:
                    break
            avgdistance.append(np.min(Ac))
            #avgdistance.append(np.sum(Ac)/compare_num)

        result_index = avgdistance.index(min(avgdistance))
    
        print('result = {}'.format(avgdistance))
        print('result index = {}'.format(result_index))
        print('Classes name with result index = {}'.format(self.dataset.get_childname(result_index)))
        
        #return result_index, self.dataset.get_childname(idx=result_index)
                        

if __name__ == '__main__':
    Face_recognition = Face(known_data_path = data_path)
    #Face_recognition.recognition()
    testimg = face_recognition.load_image_file(os.path.join(data_path, 'real__yami', '0015.jpg'))
    print(type(testimg))
    Face_recognition.recognition(Img=testimg)
    #Img_encoding = Face_recognition.get_face_encoding(Img)
    #idx, classes_name = Face_recognition.recognition(Img=testimg)

