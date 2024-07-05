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

def NaiveBayes(train_data, train_labels, test_labels, test_data):

    print('Training Naive Bayes model on dataset.')
    centroid_image = np.mean(train_data, axis=0)
    class_list = np.unique(train_labels)
    num_classes = len(class_list)
    num_pixels = train_data.shape[1]
    num_test_cases = len(test_labels)

    # Loop through every class
    prob_class_img = np.zeros( (num_classes, num_pixels) )

    for class_index in range(num_classes):

        # Create an image of average pixels for this class
        mask = train_labels==class_index
        train_data_this_class = np.compress(mask, train_data, axis=0)

        class_centroid_image = np.mean(train_data_this_class, 0)

        # Compute probability of class for each pixel
        prob_class_img[class_index] = class_centroid_image / (centroid_image+.0001) / num_classes

    # Now use the probability images to estimate the probability of each class
    # in new images
    pred = np.zeros(num_test_cases)

    # Predict all test images
    for text_index in range(num_test_cases):

        test_img = test_data[text_index]

        prob_class = []
        for classidx in range(num_classes):
            test_img_prob_class = test_img * prob_class_img[classidx]
            # Average the probabilities of all pixels
            prob_class.append( np.mean(test_img_prob_class) )

        # Pick the largest
        pred[text_index] = prob_class.index(max(prob_class))

    # Accuracy, precision & recall
    print("Accuracy:   {:.3f}".format(metrics.accuracy_score(test_labels, pred)))
    print("Precision:  {:.3f}".format(metrics.precision_score(test_labels, pred, average='weighted')))
    print("Recall:     {:.3f}".format(metrics.recall_score(test_labels, pred, average='weighted')))

    return {'accuracy': metrics.accuracy_score(test_labels, pred), 'model':None, 'pred': pred}