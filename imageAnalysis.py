import os
import timeit
import random
from datetime import datetime
import csv
import pickle

import cv2
import numpy as np

from libs.leafDiseaseDetection import LeafDisease
from libs.plot import getConfusionMatrix, plotSaveConfusionMatrix

def main():
    '''Variables'''
    dt_string = datetime.now().strftime("%Y-%m-%d %H-%M-%S")

    '''Main Params'''
    model = 'model.sav'
    datatest_dir = 'dataset/test'
    output_file = 'imageAnalysis_' + dt_string + '.csv'
    confusion_matrix_file = 'cm_' + dt_string + '.png'
    sample_feature_file = 'sampleFeature.txt'
    batch_size = 1
    lower_blue = np.array([14,32.64,22.185])
    upper_blue = np.array([34,255,232.815])
    classes = [class_name for class_name in os.listdir(datatest_dir) if os.path.isdir(os.path.join(datatest_dir, class_name).replace("\\","/"))]
    debug_index = []

    model = pickle.load(open(model, 'rb'))

    '''List Test Files'''
    test_datas = []
    img_files = []
    for i, class_name in enumerate(classes):
        debug_index.append(len(img_files))                                                      #Select 1 img each class to print it's feature
        img_files = os.listdir(os.path.join(datatest_dir, class_name).replace("\\","/"))
        for img_name in img_files:
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
    feature_file = open(sample_feature_file, 'w')
    for i, test_data in enumerate(test_datas):
        debug = (i == debug_index[0] or i == debug_index[1])
        if debug:
            print()
            print('File :', test_data[0])
        img = LeafDisease.loadImage(test_data[0])
        img_hsv_masked, glcm = LeafDisease.preprocessing(img, lower_blue, upper_blue)
        feature = LeafDisease.extractFeature(img_hsv_masked, glcm, debug=debug)

        if debug:
            feature_file.writelines('File\t\t\t: ' + test_data[0] + '\n')
            feature_file.writelines('Contrast\t\t: ' + str(feature[0]) + '\n')
            feature_file.writelines('Energy\t\t\t: ' + str(feature[1]) + '\n')
            feature_file.writelines('Homogeneity\t\t: ' + str(feature[2]) + '\n')
            feature_file.writelines('Mean\t\t\t: ' + str(feature[3]) + '\n')
            feature_file.writelines('Standard Deviation\t: ' + str(feature[4]) + '\n')
            feature_file.writelines('Variance\t\t: ' + str(feature[5]) + '\n')
            feature_file.writelines('Entropy\t\t\t: ' + str(feature[6]) + '\n')
            feature_file.writelines('Root Mean Square\t: ' + str(feature[7]) + '\n')
            feature_file.writelines('Smoothness\t\t: ' + str(feature[8]) + '\n')
            feature_file.writelines('\n')
        
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
    
    feature_file.close()
    
    '''Write to CSV'''
    with open(output_file, 'w', newline='') as csv_file:
        file_writer = csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        '''Write Header'''
        header = ['file(s)', 'file(s) size', 'total file size(Bytes)', 'time(ms)', 'prediction(s)', 'conclusion(s)']
        file_writer.writerow(header)

        '''Write Data'''
        file_writer.writerows(results)

    cm = getConfusionMatrix(y_pred=y_pred, y_true=y_test)
    plot = plotSaveConfusionMatrix(cm, classes, confusion_matrix_file)
    
if __name__ == '__main__':
    main()