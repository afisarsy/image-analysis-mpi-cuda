import os
import timeit
import random
from datetime import datetime
import csv

import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression

from libs.leafDiseaseDetection import LeafDisease
from libs.plot import getconfusionmatrix, plotconfusionmatrix

def main():
    '''Variables'''
    dt_string = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    
    '''Main Params'''
    dataset_dir = 'dataset/train'
    datatest_dir = 'dataset/test'
    output_file = 'imageAnalysisCuda_' + dt_string + '.csv'
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
    
    #random.shuffle(test_datas)

    results = []

    x_test = []
    y_test = []
    y_pred = []
    x_batch = []
    y_batch = []
    pred_batch = []

    file_paths = []
    file_sizes = []

    start = timeit.default_timer()
    batch_index = 0
    for i, test_data in enumerate(test_datas):
        hsv_img = LeafDisease.loadImage(test_data[0])
        feature = LeafDisease.extractFeature(hsv_img, lower_blue, upper_blue)
        
        file_paths.append(test_data[0][(test_data[0].find('/')+1):])
        file_sizes.append(os.path.getsize(test_data[0]))
        x_batch.append(feature)
        y_batch.append(test_data[1])
        if((i + 1) % batch_size == 0 or i == len(test_datas)-1):
            pred_batch = model.predict(x_batch)
            
            total_file_size = sum(file_sizes)
            predictions = [classes[index] for index in pred_batch]
            conclusions = [1 if pred == test_datas[(batch_index * batch_size) + index][1] else 0 for index, pred in enumerate(pred_batch)]

            exec_time = (timeit.default_timer() - start) * 1000
            result = [file_paths, file_sizes, total_file_size, exec_time, predictions, conclusions]
            results.append(result)
            print('%12i' % total_file_size, 'Bytes', '%15s' % '{0:.3f}'.format(exec_time), 'ms', file_paths)
            
            x_test += x_batch
            y_test += y_batch
            y_pred += [pred for pred in pred_batch]

            x_batch = []
            y_batch = []
            pred_batch = []
            file_paths = []
            file_sizes = []
            batch_index += 1

            start = timeit.default_timer()
    
    '''Write to CSV'''
    with open(output_file, 'w', newline='') as csv_file:
        file_writer = csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        '''Write Header'''
        header = ['file(s)', 'file(s) size', 'total file size(Bytes)', 'time(ms)', 'prediction(s)', 'conclusion(s)']
        file_writer.writerow(header)

        '''Write Data'''
        file_writer.writerows(results)

    #cm = getconfusionmatrix(y_pred=y_pred, y_true=y_test)
    #plot = plotconfusionmatrix(cm, classes)
    
if __name__ == '__main__':
    main()