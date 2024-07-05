# Softmax Regressor
# Jose Miguel Loguirato, Andres Aguilar, Maria Elena Leizaola
# CS 3368

import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

from sklearn import metrics, linear_model, preprocessing, ensemble
from pickle_helper import get_mnist_data_and_labels

def SoftMaxRegression(train_data, train_labels, test_labels, test_data):

    print('Training Soft Max Regression model on dataset.')

    model = linear_model.LogisticRegression(solver='sag', tol=1e-2, max_iter = 50) 
    model.fit(train_data, train_labels)

    # Make the class predictions
    pred = model.predict(test_data)

    # Accuracy, precision & recall
    print("Accuracy:   {:.3f}".format(metrics.accuracy_score(test_labels, pred)))
    print("Precision:  {:.3f}".format(metrics.precision_score(test_labels, pred, average='weighted')))
    print("Recall:     {:.3f}".format(metrics.recall_score(test_labels, pred, average='weighted')))

    # Per-Class Precision & Recall
    precision = metrics.precision_score(test_labels, pred, average=None)
    recall = metrics.recall_score(test_labels, pred, average=None)
    num_classes = len(np.unique(train_labels))
    """
    for n in range(num_classes):
        print("  Class {}: Precision: {:.3f} Recall: {:.3f}".format(n, precision[n], recall[n]))
    """
    return {'accuracy': metrics.accuracy_score(test_labels, pred), 'model':model, 'pred': pred}