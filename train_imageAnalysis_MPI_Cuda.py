import os
import timeit
import random
from datetime import datetime
import csv
import pickle

from mpi4py import MPI
import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression

from libs.leafDiseaseDetection import LeafDisease
from libs.imgProcessing import imgProcessing

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

def main():
    if rank == 0:
        '''Variables'''
        dt_string = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
        x_train = []
        y_train = []    

    '''Main Params'''
    dataset_dir = 'dataset/train'
    output_file = 'modelMPICuda' + '.sav'
    batch_size = 1
    lower_blue = np.array([14,32.64,22.185])
    upper_blue = np.array([34,255,232.815])

    classes = [class_name for class_name in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, class_name).replace("\\","/"))]

    for i, class_name in enumerate(classes):
        img_files = os.listdir(os.path.join(dataset_dir, class_name).replace("\\","/"))
        for j, img_name in enumerate(img_files):
            img_path = os.path.join(dataset_dir, class_name, img_name).replace("\\","/")
            if rank == 0:
                print()
                print('[', rank, ']', '({}/{})'.format(j+1, len(img_files)), 'File :', img_path)
            img = LeafDisease.loadImage(img_path)
            w, h, c = img.shape
            cropBox = imgProcessing.getCropBox(w, h, size, index=rank)
            img_crop = img[cropBox[0]:cropBox[2], cropBox[1]:cropBox[3]]
            img_hsv_masked, glcm = LeafDisease.cudaPreprocessing(img_crop, lower_blue, upper_blue)
            feature = LeafDisease.extractFeature(img_hsv_masked, glcm)
            np_feature = np.array(feature, dtype='float')

            gathered_features = None
            if rank == 0:
                gathered_features = np.empty([size, 9], dtype='float')
            comm.Gather(np_feature, gathered_features, root=0)

            if rank == 0:
                print('[', rank, ']', 'Gathered features :', gathered_features)
                combined_feature = LeafDisease.combineFeatures(gathered_features)
                print('[', rank, ']', 'Combined feature :', combined_feature)

                x_train.append(combined_feature)
                y_train.append(i)
    
    if rank == 0:
        model = LogisticRegression()
        model.fit(x_train, y_train)             # Train model
        pickle.dump(model, open(output_file, 'wb'))

if __name__ == '__main__':
    main()