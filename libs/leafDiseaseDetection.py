import os
import random

import cv2
import numpy as np
from skimage.feature import greycomatrix,greycoprops
from sklearn.metrics.cluster import entropy
from sklearn.preprocessing import StandardScaler

class LeafDisease:
    '''CPU Section'''
    '''Load Image'''
    @staticmethod
    def loadImage(img_path):
        img = cv2.imread(img_path)
        if img is None:
            print()
            print('Error :', img_path)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        return img_hsv

    '''Extract Feature'''
    @staticmethod
    def extractFeature(img_hsv, lower_mask, upper_mask, debug = False):
        mask = cv2.inRange(img_hsv, lower_mask, upper_mask)                         # Create mask
        img_hsv_masked = cv2.bitwise_and(img_hsv, img_hsv, mask=mask)               # Applying bitwise operator between mask and originl image
        
        grey_img_hsv_masked = cv2.cvtColor(img_hsv_masked, cv2.COLOR_BGR2GRAY)
        glcm = greycomatrix(grey_img_hsv_masked, [1], [0])                          # Get Grey Level Correlation Matrix

        contrast = greycoprops(glcm, 'contrast')                                    # Extract feature : contrast
        energy = greycoprops(glcm, 'energy')                                        # Extract feature : energy
        homogeneity = greycoprops(glcm, 'homogeneity')                              # Extract feature : homogeneity

        mean = img_hsv_masked.mean()                                                # Extract feature : mean
        std = img_hsv_masked.std()                                                  # Extract feature : standard deviation
        var = img_hsv_masked.var()                                                  # Extract feature : variance

        e = entropy(img_hsv_masked)                                                 # Extract feature : entropy

        rms = np.sqrt(np.mean(img_hsv_masked**2))                                   # Extract feature : root means square

        smoothness = 1 - (1 / (1 + img_hsv_masked.sum()))                           # Extract feature : smoothness

        if(debug):
            print('%-25s' % 'Contrast',':', contrast[0][0])
            print('%-25s' % 'Energy',':', energy[0][0])
            print('%-25s' % 'Homogeneity', ':', homogeneity[0][0])
            print('%-25s' % 'Mean', ':', mean)
            print('%-25s' % 'Standard Deviation', ':', std)
            print('%-25s' % 'Variance', ':', var)
            print('%-25s' % 'Entropy', ':', e)
            print('%-25s' % 'Root Mean Square', ':', rms)
            print('%-25s' % 'Image Sum', ':', img_hsv_masked.sum())
            print('%-25s' % 'Smoothness', ':', smoothness)

        return [contrast[0][0], energy[0][0], homogeneity[0][0], mean, std, var, e, rms, smoothness]
    





    '''GPU Section'''
    '''Load Image CUDA'''
    @staticmethod
    def cudaLoadImage(img_path):
        img = cv2.imread(img_path)
        if img is None:
            print()
            print('Error :', img_path)
        gpu_img = cv2.cuda_GpuMat()
        gpu_img.upload(img)
        gpu_img_hsv = cv2.cuda.cvtColor(gpu_img, cv2.COLOR_BGR2HSV)
        img_hsv = gpu_img_hsv.download()
        return img_hsv, gpu_img_hsv

    '''Extract Feature CUDA'''
    @staticmethod
    def cudaExtractFeature(img_hsv, lower_mask, upper_mask, debug = False):
        mask = cv2.inRange(img_hsv, lower_mask, upper_mask)                         # Create mask
        img_hsv_masked = cv2.bitwise_and(img_hsv, img_hsv, mask=mask)               # Applying bitwise operator between mask and originl image
        gpu_img_hsv_masked = cv2.cuda_GpuMat()
        gpu_img_hsv_masked.upload(img_hsv_masked)
        
        gpu_img_grey = cv2.cuda.cvtColor(gpu_img_hsv_masked, cv2.COLOR_BGR2GRAY)
        img_grey = gpu_img_grey.download()
        glcm = greycomatrix(img_grey, [1], [0])                                     # Get Grey Level Correlation Matrix

        contrast = greycoprops(glcm, 'contrast')                                    # Extract feature : contrast
        energy = greycoprops(glcm, 'energy')                                        # Extract feature : energy
        homogeneity = greycoprops(glcm, 'homogeneity')                              # Extract feature : homogeneity

        mean = img_hsv_masked.mean()                                                # Extract feature : mean
        std = img_hsv_masked.std()                                                  # Extract feature : standard deviation
        var = img_hsv_masked.var()                                                  # Extract feature : variance

        e = entropy(img_hsv_masked)                                                 # Extract feature : entropy

        rms = np.sqrt(np.mean(img_hsv_masked**2))                                   # Extract feature : root means square

        smoothness = 1 - (1 / (1 + img_hsv_masked.sum()))                           # Extract feature : smoothness

        if(debug):
            print()
            print('%-25s' % 'Contrast',':', contrast[0][0])
            print('%-25s' % 'Energy',':', energy[0][0])
            print('%-25s' % 'Homogeneity', ':', homogeneity[0][0])
            print('%-25s' % 'Mean', ':', mean)
            print('%-25s' % 'Standard Deviation', ':', std)
            print('%-25s' % 'Variance', ':', var)
            print('%-25s' % 'Entropy', ':', e)
            print('%-25s' % 'Root Mean Square', ':', rms)
            print('%-25s' % 'Image Sum', ':', img_hsv_masked.sum())
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
    def loadDataset(dataset_dir, mask_param, cuda=False, debug=False):
        classes = [class_name for class_name in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, class_name).replace("\\","/"))]
        dataset = []

        for i, class_name in enumerate(classes):
            images = os.listdir(os.path.join(dataset_dir, class_name).replace("\\","/"))
            for j, img_name in enumerate(images):
                img_path = os.path.join(dataset_dir, class_name, img_name).replace("\\","/")

                print('Processing  : ', img_path, '%-15s' % '({0}/{1})'.format(j+1, len(images)), end='\r')

                if not cuda:
                    img_hsv = LeafDisease.loadImage(img_path)
                    feature = LeafDisease.extractFeature(img_hsv, mask_param[0], mask_param[1], debug=debug)
                else:
                    img_hsv, gpu_img_hsv = LeafDisease.cudaLoadImage(img_path)
                    feature = LeafDisease.cudaExtractFeature(img_hsv, mask_param[0], mask_param[1], debug=debug)
                dataset.append([feature, i])
            print()
        
        random.shuffle(dataset)                             # Shuffle the data
        
        x_train = np.array([data[0] for data in dataset])
        y_train = np.array([data[1] for data in dataset])

        return (x_train, y_train), classes