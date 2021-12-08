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
            img_files = os.listdir(os.path.join(dataset_dir, class_name).replace("\\","/"))
            for img_name in img_files:
                img_path = os.path.join(dataset_dir, class_name, img_name).replace("\\","/")
                img_data.append([img_path, i])
        total_img = len(img_data)
    else:
        lower_blue = None
        upper_blue = None
        total_img = None
    
    #comm.Barrier()
    lower_blue = comm.bcast(lower_blue, root=0)
    upper_blue = comm.bcast(upper_blue, root=0)
    total_img = comm.bcast(total_img, root=0)
    #comm.Barrier()

    print('[', rank, ']', 'lower blue :', lower_blue)
    print('[', rank, ']', 'upper blue :', upper_blue)
    print('[', rank, ']', 'total img :', total_img)
    
    for i in range(total_img):
        if rank == 0:
            img = LeafDisease.loadImage(img_data[i][0])
            w, h, c = img.shape
            cropBoxes = imgProcessing.getCropBox(w, h, size)
            imgs_crop = []
            for cropBox in cropBoxes:
                imgs_crop.append(img[cropBox[0]:cropBox[2], cropBox[1]:cropBox[3]])
            imgs_crop = np.array(imgs_crop)
        else:
            imgs_crop = None
        
        img_crop = None
        comm.Scatterv(imgs_crop, img_crop, root=0)
        print('[', rank, ']', 'image :', img_crop)
        img_hsv_masked, glcm = LeafDisease.preprocessing(img_crop, lower_blue, upper_blue)
        feature = LeafDisease.extractFeature(img_hsv_masked, glcm, debug=debug)

        print('[', rank, ']', 'Feature :', features)

        sys.exit(0)

if __name__ == '__main__':
    main()