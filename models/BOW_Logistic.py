import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression

class Bow_Logistic:

    def __init__(self):
        self.arguments  = {
                    'penalty': 'l2',
                    'C': 0.3,
                    'solver': 'lbfgs',
                    'max_iter': 1e6,
                }

        print('Building Bow Logistic Regression Models ...')
        print(self.arguments)


    def fit(self, X_train, y_train, X_valid, y_valid):
        lr_bow = LogisticRegression(solver='lbfgs', max_iter=1e6, C=0.3)
        self.model = OneVsRestClassifier(lr_bow)
        self.model.fit(np.vstack([X_train, X_valid]), np.vstack([y_train, y_valid]))
        print('Finished Building Lstm Model as class attribute class.model')
        return self

    def predict(self, X):
        return self.model.decision_function(X)
