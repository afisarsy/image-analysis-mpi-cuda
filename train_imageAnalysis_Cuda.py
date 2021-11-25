import os
import timeit
import random
from datetime import datetime
import csv
import pickle

import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression

from libs.leafDiseaseDetection import LeafDisease

def main():
    '''Variables'''
    dt_string = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    
    '''Main Params'''
    dataset_dir = 'dataset/train'
    output_file = 'modelCuda' + '.sav'
    batch_size = 1
    lower_blue = np.array([14,32.64,22.185])
    upper_blue = np.array([34,255,232.815])

    (x_train, y_train), classes = LeafDisease.loadDataset(dataset_dir, [lower_blue, upper_blue], cuda=True)

    model = LogisticRegression()
    model.fit(x_train, y_train)             # Train model
    pickle.dump(model, open(output_file, 'wb'))
    
if __name__ == '__main__':
    main()