import itertools

import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

'''Calculate The Confusion Matrix'''
def getconfusionmatrix(y_pred, y_true):
    cm = confusion_matrix(y_pred=y_pred, y_true=y_true)
    return cm

'''Plot Confusion Matrix'''
def plotconfusionmatrix(cm, classes, normalize=False, title='Confusion Matrix', color_map=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=color_map)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print('Normalized Confusion Matrix')
    else:
        print('Confusion Matrix')
    print(cm)

    thresh = cm.max() / 2
    for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment='center',
            color='white' if cm[i, j] > thresh else 'black'
        )
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()