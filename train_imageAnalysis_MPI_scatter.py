import os
import timeit
import random
from datetime import datetime
import csv
import pickle
import json

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

        '''Main Params'''
        dataset_dir = 'dataset/train'
        output_file = 'modelMPIScatter' + '.sav'
        batch_size = 1
        lower_blue = np.array([14,32.64,22.185])
        upper_blue = np.array([34,255,232.815])

        classes = [class_name for class_name in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, class_name).replace("\\","/"))]
        img_data = []
        x_train = []
        y_train = []
        
        for i, class_name in enumerate(classes):
            img_files = os.listdir(os.path.join(dataset_dir, class_name).replace("\\","/"))
            for img_name in img_files:
                img_path = os.path.join(dataset_dir, class_name, img_name).replace("\\","/")
                img_data.append([img_path, i])
        total_img = len(img_data)
    else:
        lower_blue = None
        upper_blue = None
        total_img = None
    
    lower_blue = comm.bcast(lower_blue, root=0)
    upper_blue = comm.bcast(upper_blue, root=0)
    total_img = comm.bcast(total_img, root=0)
    
    for i in range(total_img):
        if rank == 0:
            print()
            print('[', rank, ']', '({}/{})'.format(i+1, total_img), 'File :', img_data[i][0])
            img = LeafDisease.loadImage(img_data[i][0])
            w, h, c = img.shape
            cropBoxes = imgProcessing.getCropBox(w, h, size)
            imgs_crop = []
            for cropBox in cropBoxes:
                cropped_image = img[cropBox[0]:cropBox[2], cropBox[1]:cropBox[3]]
                imgs_crop.append(json.dumps(cropped_image.tolist()))
        else:
            imgs_crop = None
        
        json_img = comm.scatter(imgs_crop, root=0)
        img_crop = np.array(json.loads(json_img), dtype='uint8')
        img_hsv_masked, glcm = LeafDisease.preprocessing(img_crop, lower_blue, upper_blue)
        feature = LeafDisease.extractFeature(img_hsv_masked, glcm)
        scatter_feature = np.array(feature, dtype='float')

        gathered_features = None
        if rank == 0:
            gathered_features = np.empty([size, 9], dtype='float')
        comm.Gather(scatter_feature, gathered_features, root=0)

        if rank == 0:
            print('[', rank, ']', 'Gathered features :', gathered_features)
            combined_feature = LeafDisease.combineFeatures(gathered_features)
            print('[', rank, ']', 'Combined feature :', combined_feature)

            x_train.append(combined_feature)
            y_train.append(img_data[i][1])

    if rank == 0:
        model = LogisticRegression()
        model.fit(x_train, y_train)             # Train model
        pickle.dump(model, open(output_file, 'wb'))

if __name__ == '__main__':
    main()