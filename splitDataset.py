import sys
import os
import random
import shutil
from datetime import datetime
import json

def main():
    '''Main Params'''
    output_dir = 'dataset'
    dataset_dir = '../Corn Disease detection'
    train_ratio = .6    #Train ratio = X, Test ratio = 1-X, X=(0-1)

    '''Prepare Variables'''
    classes = [class_name for class_name in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, class_name).replace("\\","/"))]
    train_dir = output_dir + '/train'
    test_dir = output_dir + '/test'

    '''Create Output Directory'''
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

        os.makedirs(train_dir)
        for class_name in classes:
            os.makedirs(train_dir +'/' + class_name)

        os.makedirs(test_dir)
        for class_name in classes:
            os.makedirs(test_dir +'/' + class_name)

    '''Load prev config'''
    prev_props = {}
    with open(os.path.join(dataset_dir, 'properties.txt')) as f:
        prev_props = json.load(f)

    '''Write properties File'''
    with open(os.path.join(output_dir, 'properties.txt'), 'w') as f:
        props = {
            'split' : {
                'source' : dataset_dir[dataset_dir.rfind('/')+1:],
                'train ratio' : train_ratio
            }
        }
        prev_props.update(props)
        f.write(json.dumps(prev_props))
    
    '''Read all dataset'''
    file_paths = []
    for root, dir, files in os.walk(dataset_dir):
        for file in files:
            file_paths.append(os.path.join(root, file).replace("\\","/")[len(dataset_dir) + 1:])
    file_paths = set(file_paths)    #Convert to set
    print('Total data :', len(file_paths))
    
    '''Split data'''

    n_train = int(len(file_paths) * train_ratio)
    train_files = random.sample(file_paths, n_train)

    test_files = file_paths.difference(train_files)
    n_val = len(val_files)

    print()
    print('Train files')
    for i, file in enumerate(train_files):
        print('({0}/{1})'.format(i+1, len(train_files)), end='\r')
        shutil.copy(os.path.join(dataset_dir, file).replace("\\","/"), os.path.join(output_dir, 'train', file).replace("\\","/"))

    print()
    print('Test files')
    for i, file in enumerate(test_files):
        print('({0}/{1})'.format(i+1, len(test_files)), end='\r')
        shutil.copy(os.path.join(dataset_dir, file).replace("\\","/"), os.path.join(output_dir, 'test', file).replace("\\","/"))

    '''Report'''
    print('Successfully splitted into :')
    print('\t', n_train, 'train data')
    print('\t', n_test, 'test data')

if __name__ == '__main__':
    main()