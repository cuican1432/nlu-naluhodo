from sklearn.tree import DecisionTreeClassifier

def pred_classes_dt(y_train_pred, y_train, y_test_pred):
    if y_train.ndim == 2:
        num_instances, num_classes = y_train.shape
    else:
        num_instances = y_train.shape[0]
        num_classes = 1
    threshold = []
    for i in range(num_classes):
        if num_classes == 1:
            clf = DecisionTreeClassifier(max_depth=1)
            clf.fit(y_train_pred[:, i].reshape(-1, 1), y_train[:, i])
            threshold += [clf.tree_.threshold[0]]
        else:
            clf = DecisionTreeClassifier(max_depth=1)
            clf.fit(y_train_pred[:, i].reshape(-1, 1), y_train[:, i])
            threshold += [clf.tree_.threshold[0]]
    pred_classes = y_test_pred >= threshold
    return pred_classes.astype(int)

from sklearn.metrics import precision_recall_curve

def pred_classes_f1(y_train_pred, y_train, y_test_pred):
    if y_train.ndim == 2:
        num_instances, num_classes = y_train.shape
    else:
        num_instances = y_train.shape[0]
        num_classes = 1
    threshold = []
    for i in range(num_classes):
        if num_classes == 1:
            precision, recall, thresholds = precision_recall_curve(y_train, train_pred)
            f1 = 2 * precision * recall / (precision + recall)
            threshold += [thresholds[np.argmax(f1)]]
        else:
            precision, recall, thresholds = precision_recall_curve(y_train[:, i], train_pred[:, i])
            f1 = 2 * precision * recall / (precision + recall)
            threshold += [thresholds[np.argmax(f1)]]
    pred_classes = y_test_pred >= threshold
    return pred_classes.astype(int)
