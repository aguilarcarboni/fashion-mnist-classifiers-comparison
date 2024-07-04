# Project 1
# Jose Miguel Loguirato, Andres Aguilar, Maria Elena Leizaola
# CS 3368

import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

from sklearn import metrics, linear_model, preprocessing, ensemble
from pickle_helper import get_mnist_data_and_labels

"""
Step 1: Pre-process the raw data files and convert them into a format suitable for Scikit-Learn classifiers
"""

# Load the training and test data from the Pickle file (or from other file if Pickle file does not exist)
if (os.path.exists("mnist_dataset.pickle")):

    print("Reading pickle file containing data")
    with open("mnist_dataset.pickle", "rb") as f:
        train_data, train_labels, test_data, test_labels = pickle.load(f)
else:

    print("Reading training dataset")
    train_data, train_labels = get_mnist_data_and_labels("train-images.idx3-ubyte", "train-labels.idx1-ubyte")
    train_size = train_data.shape[0]

    print("Reading test dataset")
    test_data, test_labels = get_mnist_data_and_labels("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte")
    test_size = test_data.shape[0]

# Standarize scale
max_value = train_data.max()
train_data = train_data / max_value
test_data = test_data / max_value

# Print some information about the training dataset
print("Preprocessing complete.")
print("Training dataset size: ", train_data.shape)
print("Test dataset size: ", test_data.shape)

"""
Step 2: Explore the dataset (quantity of examples of each class, distribution of pixel values, centroid images: overall and per-class)
"""

# Quantity of examples of each class
fig, ax = plt.subplots()
bar_colors = ['tab:red', 'tab:blue']

hist, bins = np.histogram(train_labels, bins=20)
center = (bins[:-1] + bins[1:]) / 2
width = np.diff(bins)
ax.bar(center, hist, width=width, color=bar_colors)
plt.show()


# Distribution of pixel values
hist, bins = np.histogram(train_data, bins=50)
center = (bins[:-1] + bins[1:]) / 2
width = np.diff(bins)
plt.bar(center, hist, align='center', width=width, color='red')
plt.show()

# Compute and show overall centroid image
centroid_image = np.mean(train_data, axis=0)
plt.figure()
plt.imshow(centroid_image.reshape(28, 28), cmap='gray')
plt.title('Overall Centroid Image')
plt.axis('off')
plt.show()

# Compute and show one centroid image per class
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
fig.suptitle('Centroid Images by Class')

class_list = np.unique(train_labels)
num_classes = len(class_list)
for class_index in range(num_classes):
    
    # Create an image of average pixels for this class
    mask = train_labels==class_index
    train_data_this_class = np.compress(mask, train_data, axis=0)

    class_centroid_image = np.mean(train_data_this_class, 0)
    class_centroid_image = class_centroid_image.reshape(28, 28)

    ax = axes[class_index // 5, class_index % 5]
    ax.imshow(class_centroid_image, cmap='gray')
    ax.set_title(f'Class {class_index}')
    ax.axis('off')

plt.tight_layout()
plt.show()

"""
Step 3: Attempt multiple classifiers (at least 4).  You are free to choose which ones.  
You should attempt to optimize the available parameters of each classifier to get the best results.
"""

def NaiveBayes():

    print('Training Naive Bayes model on dataset.')
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

def RidgeRegression():
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

def SoftMaxRegression():

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

def RandomForest():

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

def findFeatureImportances(best_model):

    print(best_model['model'])

    # Display the feature importances as an image
    coef_img = best_model['model'].feature_importances_.reshape(28, 28)
    plt.figure()
    plt.imshow(coef_img, cmap="gray_r")
    plt.show()

    num_displayed = 0
    x = 0
    while (num_displayed < 10):
        x += 1

        # Skip correctly predicted 
        if (model['pred'][x] == test_labels[x]):
            continue

        num_displayed += 1

        # Display the images
        image = test_data[x].reshape(28,28)
        plt.figure()
        plt.imshow(image, cmap="gray_r")
        plt.title("Predicted: "+str(model['pred'][x])+" Correct: "+str(test_labels[x]))
        plt.show()

centroid_image = centroid_image.reshape(784)
best_model = {'accuracy': 0, 'model':None, 'pred': None}

model = NaiveBayes()
if model['accuracy'] > best_model['accuracy']:
    best_model = model

model = RidgeRegression()
if model['accuracy'] > best_model['accuracy']:
    best_model = model

model = SoftMaxRegression()
if model['accuracy'] > best_model['accuracy']:
    best_model = model

model = RandomForest()
if model['accuracy'] > best_model['accuracy']:
    best_model = model

print(best_model)

findFeatureImportances(best_model)