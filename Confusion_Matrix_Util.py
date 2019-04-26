import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_test, y_pred, title=None, x_classes=None, y_classes=None):
    if y_pred.ndim != 1:
        print("y_pred don't accept probability, and please don't binarize")
    cm = confusion_matrix(y_test, y_pred)
    num_classes = cm.shape[0]
    count = np.unique(y_test, return_counts=True)[1].reshape(num_classes, 1)
    if 0 in count:
        print('Detect 0 value in True Label, please double check')

    fig = plt.figure(figsize=(10,6))
    ax = plt.subplot(111)
    im = ax.imshow(cm/count, cmap='YlGnBu')
    im.set_clim(0, 1)
    cbar = ax.figure.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    if type(x_classes) == list:
        ax.set_xticklabels(x_classes, set_rotation(45))
    if type(y_classes) == list:
        ax.set_yticklabels(y_classes)
    plt.yticks(fontsize=13)
    plt.xticks(fontsize=13)
    for i in range(num_classes):
        for j in range(num_classes):
            text = ax.text(i, j, cm[j][i], ha="center", va="center", color="w" if (cm/count)[j, i] > 0.5 else "black", fontsize=13)
    ax.set_ylabel('True Label', fontsize=16)
    ax.set_xlabel('Predicted Label', fontsize=16)
    ax.set_title('Confusion Matrix for Coarse Genre', fontsize=16, fontweight='bold')
    plt.show()
