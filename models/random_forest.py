# Softmax Regressor
# Jose Miguel Loguirato, Andres Aguilar, Maria Elena Leizaola
# CS 3368

import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

from sklearn import metrics, linear_model, preprocessing, ensemble
from mnist_helper import get_mnist_data_and_labels

def RandomForest(train_data, train_labels, test_labels, test_data):

    print('Training Random Forest Regression model on dataset.')
    
    model = ensemble.RandomForestClassifier(n_estimators = 100, min_samples_leaf = 1e-4) 

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