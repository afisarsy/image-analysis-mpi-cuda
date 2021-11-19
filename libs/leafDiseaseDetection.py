import os
import random

import cv2
import numpy as np
from skimage.feature import greycomatrix,greycoprops
from sklearn.metrics.cluster import entropy
from sklearn.preprocessing import StandardScaler

class LeafDisease:
    '''Load Image'''
    @staticmethod
    def loadImage(image_path):
        img = cv2.imread(image_path)
        hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        return hsv_image

    '''Extract Feature'''
    @staticmethod
    def extractFeature(hsv_img, lower_mask, upper_mask, debug = False):
        mask = cv2.inRange(hsv_img, lower_mask, upper_mask)                         # Create mask
        masked_hsv_img = cv2.bitwise_and(hsv_img, hsv_img, mask=mask)               # Applying bitwise operator between mask and originl image
        
        grey_masked_hsv_img = cv2.cvtColor(masked_hsv_img, cv2.COLOR_BGR2GRAY)
        glcm = greycomatrix(grey_masked_hsv_img, [1], [0])                          # Get Grey Level Correlation Matrix

        contrast = greycoprops(glcm, 'contrast')                                    # Extract feature : contrast
        energy = greycoprops(glcm, 'energy')                                        # Extract feature : energy
        homogeneity = greycoprops(glcm, 'homogeneity')                              # Extract feature : homogeneity

        mean = masked_hsv_img.mean()                                                # Extract feature : mean
        std = masked_hsv_img.std()                                                  # Extract feature : standard deviation
        var = masked_hsv_img.var()                                                  # Extract feature : variance

        e = entropy(masked_hsv_img)                                                 # Extract feature : entropy

        rms = np.sqrt(np.mean(masked_hsv_img**2))                                   # Extract feature : root means square

        smoothness = 1 - (1 / (1 + masked_hsv_img.sum()))                           # Extract feature : smoothness

        if(debug):
            print('%-25s' % 'Contrast',':', contrast[0][0])
            print('%-25s' % 'Energy',':', energy[0][0])
            print('%-25s' % 'Homogeneity', ':', homogeneity[0][0])
            print('%-25s' % 'Mean', ':', mean)
            print('%-25s' % 'Standard Deviation', ':', std)
            print('%-25s' % 'Variance', ':', var)
            print('%-25s' % 'Entropy', ':', e)
            print('%-25s' % 'Root Mean Square', ':', rms)
            print('%-25s' % 'Image Sum', ':', masked_hsv_img.sum())
            print('%-25s' % 'Smoothness', ':', smoothness)

        return [contrast[0][0], energy[0][0], homogeneity[0][0], mean, std, var, e, rms, smoothness]

    '''Scale Data'''
    @staticmethod
    def scaleData(x, scaler = None):
        if scaler is None:
            scaler = StandardScaler()
            scaler.fit(x)
        
        return scaler.transform(x)
    
    '''Load Dataset'''
    @staticmethod
    def loadDataset(dataset_dir, mask_param):
        classes = [class_name for class_name in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, class_name).replace("\\","/"))]
        dataset = []

        for i, class_name in enumerate(classes):
            for img_name in os.listdir(os.path.join(dataset_dir, class_name).replace("\\","/")):
                img_path = os.path.join(dataset_dir, class_name, img_name).replace("\\","/")
                hsv_img = LeafDisease.loadImage(img_path)
                feature = LeafDisease.extractFeature(hsv_img, mask_param[0], mask_param[1])
                dataset.append([feature, i])
        
        random.shuffle(dataset)                             # Shuffle the data
        
        x_train = np.array([data[0] for data in dataset])
        y_train = np.array([data[1] for data in dataset])

        return (x_train, y_train), classes