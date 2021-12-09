import os
import timeit
import random
from datetime import datetime
import csv
import pickle

from mpi4py import MPI
import cv2
import numpy as np

from libs.imgProcessing import imgProcessing
from libs.leafDiseaseDetection import LeafDisease
from libs.plot import getConfusionMatrix, plotSaveConfusionMatrix

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

def main():
    '''Variables'''
    dt_string = datetime.now().strftime("%Y-%m-%d %H-%M-%S")

    '''Main Params'''
    model = 'modelMPI.sav'
    datatest_dir = 'dataset/test'
    output_file = 'imageAnalysisMPI_' + dt_string + '.csv'
    confusion_matrix_file = 'cmMPI_' + dt_string + '.png'
    sample_feature_file = 'sampleFeatureMPI.txt'
    batch_size = 1
    lower_blue = np.array([14,32.64,22.185])
    upper_blue = np.array([34,255,232.815])

    classes = [class_name for class_name in os.listdir(datatest_dir) if os.path.isdir(os.path.join(datatest_dir, class_name).replace("\\","/"))]

    if rank == 0:
        debug_index = []
        model = pickle.load(open(model, 'rb'))

    '''List Test Files'''
    test_datas = []
    img_files = []
    for i, class_name in enumerate(classes):
        if rank == 0:
            debug_index.append(len(img_files))                                                      #Select 1 img each class to print it's feature
        img_files = os.listdir(os.path.join(datatest_dir, class_name).replace("\\","/"))
        for img_name in img_files:
            img_path = os.path.join(datatest_dir, class_name, img_name).replace("\\","/")
            test_datas.append([img_path, i])
    
    #random.shuffle(test_datas)

    if rank == 0:
        results = []

        x_test = []
        y_test = []
        y_pred = []
        x_batch = []
        y_batch = []
        pred_batch = []
        batch_index = 0

        file_paths = []
        file_sizes = []

        start = timeit.default_timer()
        feature_file = open(sample_feature_file, 'w')

    for i, test_data in enumerate(test_datas):
        if rank == 0:
            debug = (i == debug_index[0] or i == debug_index[1])
            if debug:
                print()
                print('[', rank, ']', '({}/{})'.format(i+1, len(test_datas)), 'File :', test_data[0])

        img = LeafDisease.loadImage(test_data[0])
        w, h, c = img.shape
        cropBox = imgProcessing.getCropBox(w, h, size, index=rank)
        img_crop = img[cropBox[0]:cropBox[2], cropBox[1]:cropBox[3]]
        img_hsv_masked, glcm = LeafDisease.preprocessing(img_crop, lower_blue, upper_blue)
        feature = LeafDisease.extractFeature(img_hsv_masked, glcm)
        np_feature = np.array(feature, dtype='float')

        gathered_features = None
        if rank == 0:
            gathered_features = np.empty([size, 9], dtype='float')
        comm.Gather(np_feature, gathered_features, root=0)

        if rank == 0:
            contrast = max([feature[0] for feature in gathered_features])
            energy = min([feature[1] for feature in gathered_features])
            homogeneity = min([feature[2] for feature in gathered_features])
            mean = max([feature[3] for feature in gathered_features])
            std = max([feature[4] for feature in gathered_features])
            var = max([feature[5] for feature in gathered_features])
            entropy = max([feature[6] for feature in gathered_features])
            rms = max([feature[7] for feature in gathered_features])
            smoothness = max([feature[8] for feature in gathered_features])
            combined_feature = [contrast, energy, homogeneity, mean, std, var, entropy, rms, smoothness]
            print('[', rank, ']', 'Combined feature :', combined_feature)

            if debug:
                feature_file.writelines('File\t\t\t: ' + test_data[0] + '\n')
                feature_file.writelines('Contrast\t\t: ' + str([ftr[0] for ftr in gathered_features]) + '\n')
                feature_file.writelines('Energy\t\t\t: ' + str([ftr[1] for ftr in gathered_features]) + '\n')
                feature_file.writelines('Homogeneity\t\t: ' + str([ftr[2] for ftr in gathered_features]) + '\n')
                feature_file.writelines('Mean\t\t\t: ' + str([ftr[3] for ftr in gathered_features]) + '\n')
                feature_file.writelines('Standard Deviation\t: ' + str([ftr[4] for ftr in gathered_features]) + '\n')
                feature_file.writelines('Variance\t\t: ' + str([ftr[5] for ftr in gathered_features]) + '\n')
                feature_file.writelines('Entropy\t\t\t: ' + str([ftr[6] for ftr in gathered_features]) + '\n')
                feature_file.writelines('Root Mean Square\t: ' + str([ftr[7] for ftr in gathered_features]) + '\n')
                feature_file.writelines('Smoothness\t\t: ' + str([ftr[8] for ftr in gathered_features]) + '\n')
                feature_file.writelines('Combined Contrast\t\t: ' + str(contrast) + '\n')
                feature_file.writelines('Combined Energy\t\t\t: ' + str(energy) + '\n')
                feature_file.writelines('Combined Homogeneity\t\t: ' + str(homogeneity) + '\n')
                feature_file.writelines('Combined Mean\t\t\t: ' + str(mean) + '\n')
                feature_file.writelines('Combined Standard Deviation\t: ' + str(std) + '\n')
                feature_file.writelines('Combined Variance\t\t: ' + str(var) + '\n')
                feature_file.writelines('Combined Entropy\t\t\t: ' + str(entropy) + '\n')
                feature_file.writelines('Combined Root Mean Square\t: ' + str(rms) + '\n')
                feature_file.writelines('Combined Smoothness\t\t: ' + str(smoothness) + '\n')
                feature_file.writelines('\n')
            
            file_paths.append(test_data[0][(test_data[0].find('/')+1):])
            file_sizes.append(os.path.getsize(test_data[0]))
            x_batch.append(combined_feature)
            y_batch.append(test_data[1])
            if((i + 1) % batch_size == 0 or i == len(test_datas)-1):
                pred_batch = model.predict(x_batch)
                
                total_file_size = sum(file_sizes)
                predictions = [classes[index] for index in pred_batch]
                conclusions = [1 if pred == test_datas[(batch_index * batch_size) + index][1] else 0 for index, pred in enumerate(pred_batch)]

                exec_time = (timeit.default_timer() - start) * 1000
                result = [file_paths, file_sizes, total_file_size, exec_time, predictions, conclusions]
                results.append(result)
                print('[', rank, ']', '%12i' % total_file_size, 'Bytes', '%15s' % '{0:.3f}'.format(exec_time), 'ms', file_paths)
                print()
                
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
    
    if rank == 0:
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