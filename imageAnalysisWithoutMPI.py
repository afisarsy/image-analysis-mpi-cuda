import os
import timeit
import random

import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression

from libs.leafDiseaseDetection import LeafDisease
from libs.plot import getconfusionmatrix, plotconfusionmatrix

def main():
    '''Main Params'''
    dataset_dir = 'dataset/train'
    datatest_dir = 'dataset/test'
    batch_size = 1
    lower_blue = np.array([14,32.64,22.185])
    upper_blue = np.array([34,255,232.815])

    (x_train, y_train), classes = LeafDisease.loadDataset(dataset_dir, [lower_blue, upper_blue])
    #print(x_train)
    #print(y_train)
    #print(x_test)
    #print(y_test)
    #print(classes)

    model = LogisticRegression()
    model.fit(x_train, y_train)             # Train model
    #print(model)

    '''List Test Files'''
    test_datas = []
    for i, class_name in enumerate(classes):
        for img_name in os.listdir(os.path.join(datatest_dir, class_name).replace("\\","/")):
            img_path = os.path.join(datatest_dir, class_name, img_name).replace("\\","/")
            test_datas.append([img_path, i])
    
    random.shuffle(test_datas)

    x_test = []
    y_test = []
    y_pred = []
    x_batch = []
    y_batch = []
    pred_batch = []
    start = timeit.default_timer()
    for i, test_data in enumerate(test_datas):
        hsv_img = LeafDisease.loadImage(test_data[0])
        feature = LeafDisease.extractFeature(hsv_img, lower_blue, upper_blue)
            
        x_batch.append(feature)
        y_batch.append(test_data[1])
        if((i + 1) % batch_size == 0 or i == len(test_datas)-1):
            pred_batch = model.predict(x_batch)
            stop = timeit.default_timer()

            if batch_size > 1:
                print('{0:.3f}'.format((stop - start) * 1000), 'ms', 'Predictions :', [classes[index] for index in pred_batch], 'Conclusion :', ['Correct' if pred == test_datas[index][1] else 'Wrong' for index, pred in enumerate(pred_batch)])
            else:
                print('{0:.3f}'.format((stop - start) * 1000), 'ms', 'Predictions :', [classes[index] for index in pred_batch], 'Conclusion :', ['Correct' if pred == test_datas[i][1] else 'Wrong' for pred in pred_batch])

            x_test += x_batch
            y_test += y_batch
            y_pred += [pred for pred in pred_batch]

            x_batch = []
            y_batch = []
            pred_batch = []

            start = timeit.default_timer()

    #cm = getconfusionmatrix(y_pred=y_pred, y_true=y_test)
    #plot = plotconfusionmatrix(cm, classes)
    
if __name__ == '__main__':
    main()