# Softmax Regressor
# Jose Miguel Loguirato, Andres Aguilar, Maria Elena Leizaola
# CS 3368

import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

from sklearn import preprocessing, metrics, linear_model

def RidgeRegression(train_data, train_labels, test_labels, test_data):
    print('Training Ridge Regression model on dataset.')

    # One hot encode the data
    encoder = preprocessing.OneHotEncoder(categories='auto', sparse_output=False)
    train_labels_onehot = encoder.fit_transform(train_labels.reshape(-1, 1))

    test_labels_onehot = encoder.transform(test_labels.reshape(-1, 1))
    num_classes = len(encoder.categories_[0])

    # Train a linear regression classifier
    model = linear_model.Ridge(alpha=0.5)
    model.fit(train_data, train_labels_onehot)

    # Predict the probabilities of each class
    pred_proba = model.predict(test_data)

    # Pick the maximum
    pred = np.argmax(pred_proba, axis=1).astype("uint8")

    # Accuracy, precision & recall
    print("Accuracy:   {:.3f}".format(metrics.accuracy_score(test_labels, pred)))
    print("Precision:  {:.3f}".format(metrics.precision_score(test_labels, pred, average='weighted')))
    print("Recall:     {:.3f}".format(metrics.recall_score(test_labels, pred, average='weighted')))

    # Per-Class Precision & Recall
    """
    precision = metrics.precision_score(test_labels, pred, average=None)
    recall = metrics.recall_score(test_labels, pred, average=None)
    for n in range(num_classes):
        print("  Class {}: Precision: {:.3f} Recall: {:.3f}".format(n, precision[n], recall[n]))
    """
    return {'accuracy': metrics.accuracy_score(test_labels, pred), 'model':model, 'pred': pred}