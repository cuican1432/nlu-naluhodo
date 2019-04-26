import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from scipy import interp
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

def plot_roc_curve(y_test, y_pred, title=None, micro=False, macro=True, per_class=False):

    if y_train.ndim == 2:
        num_instances, num_classes = y_train.shape
    else:
        num_instances = y_train.shape[0]
        num_classes = 1
    if (num_classes != 2) and (y_test.ndim == 1):
        bi_y_test = label_binarize(y_test, classes=range(num_classes))
    else:
        bi_y_test = y_test
    
    fpr = {}
    tpr = {}
    roc_auc = {}
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(bi_y_test[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    fpr['micro'], tpr['micro'], _ = roc_curve(y_test.ravel(), y_pred.ravel())
    roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])

    # Compute macro-average ROC curve and AUC
    # Aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))
    # Interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    # Average and compute AUC
    mean_tpr /= num_classes

    fpr['macro'] = all_fpr
    tpr['macro'] = mean_tpr
    roc_auc['macro'] = auc(fpr['macro'], tpr['macro'])

    # Plot all ROC curves
    plt.figure(figsize=(10, 10))
    if micro == True:
        plt.plot(fpr['micro'], tpr['micro'],
                 label='micro-average ROC curve (area = {0:0.4f})'
                       ''.format(roc_auc['micro']),
                 color='orangered', linestyle=':', linewidth=3)

    if macro == True:
        plt.plot(fpr['macro'], tpr['macro'],
                 label='macro-average ROC curve (area = {0:0.4f})'
                       ''.format(roc_auc['macro']),
                 color='navy', linestyle=':', linewidth=3)

    if per_class == True:
        for i in range(num_classes):
            plt.plot(fpr[i], tpr[i], alpha=0.2,
                     label='ROC curve of class {0} (area = {1:0.4f})'
                     ''.format(i+1, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xticks(fontsize=13)
    plt.xticks(fontsize=13)
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    if type(title) == str:
        plt.title(title, fontsize=16)
    elif title != None:
        print('Title must be a string.')
        plt.title('ROC Curves', fontsize=16)
    else:
        plt.title('ROC Curves', fontsize=16)
    plt.legend(loc=4)
    plt.show()
