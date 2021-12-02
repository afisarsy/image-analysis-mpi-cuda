import os
import timeit
import random
from datetime import datetime
import csv
import pickle
import sys

from mpi4py import MPI
import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression

from libs.leafDiseaseDetection import LeafDisease
from libs.imgProcessing import imgProcessing

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

def master():
    model = LogisticRegression()
    model.fit(x_train, y_train)             # Train model
    pickle.dump(model, open(output_file, 'wb'))

def main():
    if rank == 0:
        '''Variables'''
        dt_string = datetime.now().strftime("%Y-%m-%d %H-%M-%S")

        '''Main Params'''
        dataset_dir = 'dataset/train'
        output_file = 'model' + '.sav'
        batch_size = 1
        lower_blue = np.array([14,32.64,22.185])
        upper_blue = np.array([34,255,232.815])

        classes = [class_name for class_name in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, class_name).replace("\\","/"))]
        img_data = []
        
        for i, class_name in enumerate(classes):
            images = os.listdir(os.path.join(dataset_dir, class_name).replace("\\","/"))
            for img_name in images:
                img_path = os.path.join(dataset_dir, class_name, img_name).replace("\\","/")
                img_data.append([img_path, i])
        total_img = len(img_data)
    else:
        lower_blue = None
        upper_blue = None
        total_img = None
    
    comm.bcast(lower_blue, root=0)
    comm.bcast(upper_blue, root=0)
    comm.bcast(total_img, root=0)
    comm.Barrier()
    
    for i in range(total_img):
        if rank == 0:
            img_hsv = LeafDisease.loadImage(img_data[i][0])
            w, h = img_hsv.shape
            cropBoxes = imgProcessing.getCropBox(w, h, size)
            imgs_hsv_crop = []
            for cropBox in cropBoxes:
                imgs_hsv_crop.append(img_hsv[cropBox[0]:cropBox[2], cropBox[1]:cropBox[3]])
            img_hsv_crop = None
        else:
            img_hsv_crop = None
        
        comm.scatter(imgs_hsv_crop, img_hsv_crop, root=0)

        features = LeafDisease.extractFeature(img_hsv_crop, lower_blue, upper_blue)
        if rank == 0:
            print('(Master)', 'File :', img_data[i][0], 'Feature :', features)
        else:
            print('(Slave-{})'.format(rank), 'File :', img_data[i][0], 'Feature :', features)
        sys.exit(0)

if __name__ == '__main__':
    main()