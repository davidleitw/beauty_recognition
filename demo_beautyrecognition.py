import os 
from scipy import misc
import tensorflow as tf
import detect_face
import facenet
import pickle
import cv2 
import numpy as np
import time 

Traindata_dir = r'./train_img'
Testdata_dir = r'./test_data'
Model_dir = r'./model/20170511-185253.pb'
Classifier = r'./class/classifier.pkl'
Npy = r'./npy'

class Testing_tool(object):
    def __init__(self, train_dir=None, test_dir=None, model_dir=None, classifier=None, npy=None):
        self.train_dir = os.path.expanduser(train_dir)
        self.test_dir = os.path.expanduser(test_dir)
        self.model_dir = os.path.expanduser(model_dir)
        self.classifier = os.path.expanduser(classifier)
        self.npy = os.path.expanduser(npy)
        
        # each class name
        self.Classname = os.listdir(self.train_dir)
        self.Classname.sort()
        
        testimglist = os.listdir(self.test_dir)
        # print('test img list type = {}'.format(type(testimglist)))
        # testimglist.sort()
        
        # self.testdata_imgname = [Imgpath for Imgpath in os.path.join(self.test_dir, testimglist)]
        self.testdata_imgname = []
        self.testdata = []
        for img_name in testimglist:
            self.testdata_imgname.append(img_name)
            Img = cv2.imread(os.path.join(Testdata_dir, img_name), 0)
            self.testdata.append(Img)
            # print('type of {}, shape of {}'.format(type(Img), Img.shape))

        self.test_analysis = []    
        
        self.min_size = 20
        self.threshold = [0.6, 0.7, 0.7]
        self.factor = 0.709
        self.margin = 44
        self.frame_interval = 3
        self.batchsize = 1000
        self.image_size = 182
        self.input_image_size = 160
        self.Image_show = True
        self.Output_analysis = True
        self.file_write = open('data_analysis.txt', 'a+')

    def recognition_testdata(self):
            with tf.Graph().as_default():
                gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
                sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
                with sess.as_default():
                    pnet, rnet, onet = detect_face.create_mtcnn(sess, self.npy)
                    facenet.load_model(self.model_dir)

                    images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                    embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                    phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

                    embedding_size = embeddings.get_shape()[1]

                    self.classifier = os.path.expanduser(self.classifier)
                    with open(self.classifier, 'rb') as infile:
                        (model, classname) = pickle.load(infile)
                    for idx, img in enumerate(self.testdata):
                        print(idx)
                        print('img shape = {}'.format(img.shape))
                        c = 0

                        frame = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
                        print('frame shape = {}'.format(frame.shape))
                        timeF = self.frame_interval

                        if(c % timeF == 0):

                            if frame.ndim == 2:
                                frame = facenet.to_rgb(frame)

                            frame = frame[:, :, 0:3]
                            bounding_box, _ = detect_face.detect_face(frame, self.min_size,
                                                                      pnet, rnet, onet,
                                                                      self.threshold, self.factor)
                            # print(bounding_box)
                            num_of_faces = bounding_box.shape[0]
                            print('num of faces = {}'.format(num_of_faces))
                                # mean we get the face
                            if num_of_faces > 0:
                                det = bounding_box[:, 0:4]
                                imgsize = np.asarray(frame.shape)[0:2]

                                cropped = []
                                scaled  = []
                                scaled_reshape = []
                                b = np.zeros((num_of_faces, 4), dtype=np.int32)

                                for i in range(num_of_faces):

                                    emb_array = np.zeros((1, embedding_size))
                                    for _idx in range(4):
                                        b[i][_idx] = det[i][_idx]

                                    if b[i][0] <= 0 or b[i][1] <= 0 or b[i][2] >= len(frame[0]) or b[i][3] >= len(frame[0]):
                                        print('face is to close')
                                        continue

                                    cropped.append(frame[b[i][1]:b[i][3], b[i][0]:b[i][2], :])
                                    cropped[i] = facenet.flip(cropped[i], False)
                                    scaled.append(misc.imresize(cropped[i], (self.image_size, self.image_size), interp='bilinear'))
                                    scaled[i] = cv2.resize(scaled[i], (self.input_image_size, self.input_image_size), interpolation=cv2.INTER_CUBIC)
                                    scaled[i] = facenet.prewhiten(scaled[i])
                                    scaled_reshape.append(scaled[i].reshape(-1, self.input_image_size, self.input_image_size, 3))
                                    feed_dict = {images_placeholder: scaled_reshape[i], phase_train_placeholder: False}
                                    emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict)
                                    predictions = model.predict_proba(emb_array) # it's a list show each face %
                                    print('predictions for each face = {}'.format(predictions))
                                    print('predictions shape = {}'.format(predictions.shape))
                                    best_class_indices = np.argmax(predictions, axis=1) # find the best one its index
                                    print('b_i = {}'.format(best_class_indices))
                                    best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                                    print('best_class_probabilities = {}'.format(best_class_probabilities))
                                    # about rectangle, i have some better idea.
                                    cv2.rectangle(frame, (b[i][0], b[i][1]), (b[i][2], b[i][3]), (0, 255, 0), 2)    # boxing face
                                    #

                                    text_x = b[i][0]
                                    text_y = b[i][3] + 20
                                    # print('results of index = {}'.format(best_class_indices))
                                    # print('result of the name is {}'.format(self.classname[best_class_indices]))
                                    # put the text to show who in the image.
                                    cv2.putText(frame, self.Classname[best_class_indices[0]], (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), thickness=1, lineType=2)
                                    '''                            
                                    for Human in self.Classname:
                                        if self.Classname[best_class_indices[0]] == Human:
                                            result_name = self.Classname[best_class_indices[0]]
                                            cv2.putText(frame, self.Classname[best_class_indices], (text_x, text_y),
                                                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), thickness=1,
                                                        lineType=2)
                                    '''

                                    # if self.Image_show is True:
                                        # cv2.imshow('Image {}'.format(idx+1), frame)
                                        # if cv2.waitkey(10000000) & 0xFF == ord('q'):
                                            # cv2.destroyAllWindows()

                                    if self.Output_analysis is True:
                                        print('idx = {}'.format(idx))
                                        self.test_analysis.append('Information')
                                        self.file_write.write('Image name: {}\nafter recognition, the people who in the image is {}\n\n'.format(self.testdata_imgname[idx],
                                                                                                                                                self.Classname[best_class_indices[0]]))
    def set_Image_show(self, flag=True):
        self.Image_show = flag
    
    def set_Output_analysis(self, flag=True):
        self.Output_analysis = flag

    def Test_data(self):
        for img in self.testdata:
            print(img)

if __name__ == '__main__':
    Test_tool = Testing_tool(Traindata_dir, Testdata_dir, Model_dir, Classifier, Npy)
    #  Test_tool.Test_data()
    Test_tool.recognition_testdata()
    
