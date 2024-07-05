# Project 1
# Jose Miguel Loguirato, Andres Aguilar, Maria Elena Leizaola
# CS 3368

import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

from sklearn import metrics, linear_model, preprocessing, ensemble
import skimage

from pickle_helper import get_mnist_data_and_labels


from naive_bayes import NaiveBayes
from ridge import RidgeRegression
from softmax import SoftMaxRegression
from random_forest import RandomForest

def findFeatureImportances(best_model):

    num_displayed = 0
    x = 0

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle('10 Misclassifications')

    while (num_displayed < 10):
        x += 1

        # Skip correctly predicted 
        if (model['pred'][x] == test_labels[x]):
            continue

        # Display the images
        image = test_data[x].reshape(28,28)
        ax = axes[num_displayed // 5, num_displayed % 5]
        ax.imshow(image, cmap='gray')
        ax.set_title("Predicted: "+str(model['pred'][x])+" Correct: "+str(test_labels[x]))
        ax.axis('off')

        num_displayed += 1

    plt.tight_layout()
    plt.show()

    # Display the feature importances as an image
    coef_img = best_model['model'].feature_importances_.reshape(28, 28)
    plt.figure()
    plt.title('Feature importances as an image.')
    plt.imshow(coef_img, cmap="gray_r")
    plt.show()

"""
Step 1: Pre-process the raw data files and convert 
them into a format suitable for Scikit-Learn classifiers
"""

# Load the training and test data from the Pickle file (or from other file if Pickle file does not exist)
if (os.path.exists("fashion/dataset.pickle")):

    print("Reading pickle file containing data")
    with open("fashion/dataset.pickle", "rb") as f:
        train_data, train_labels, test_data, test_labels = pickle.load(f)

else:

    print("Reading training dataset")
    train_data, train_labels = get_mnist_data_and_labels("fashion/train-images.idx3-ubyte", "fashion/train-labels.idx1-ubyte")
    train_size = train_data.shape[0]

    print("Reading test dataset")
    test_data, test_labels = get_mnist_data_and_labels("fashion/t10k-images.idx3-ubyte", "fashion/t10k-labels.idx1-ubyte")
    test_size = test_data.shape[0]

    # Archive the data into a pickle file
    print('Dumping Pickle file')
    with open("fashion/dataset.pickle", 'wb') as f:
        pickle.dump([train_data, train_labels, test_data, test_labels],f, pickle.HIGHEST_PROTOCOL)
    
    print("Complete")

# Standarize scale
max_value = train_data.max()
train_data = train_data / max_value
test_data = test_data / max_value

# Print some information about the training dataset
print("Preprocessing complete.")
print("Training dataset size: ", train_data.shape)
print("Test dataset size: ", test_data.shape)

"""
Step 2: Explore the dataset (quantity of examples of each class, 
distribution of pixel values, centroid images: overall and per-class)
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

best_model = {'accuracy': 0, 'model':None, 'pred': None}

"""
model = NaiveBayes(train_data, train_labels, test_labels, test_data)
if model['accuracy'] > best_model['accuracy']:
    best_model = model

model = RidgeRegression(train_data, train_labels, test_labels, test_data)
if model['accuracy'] > best_model['accuracy']:
    best_model = model

model = SoftMaxRegression(train_data, train_labels, test_labels, test_data)
if model['accuracy'] > best_model['accuracy']:
    best_model = model
"""
    
results = []
model = RandomForest(train_data, train_labels, test_labels, test_data)
if model['accuracy'] > best_model['accuracy']:
    best_model = model

print('Best model:', best_model)
results.append(best_model['accuracy'])
findFeatureImportances(best_model)
  
def resizeData(original_size, target_size, data):

    length = data.shape[0]

    # Resize all the training images
    data_resized = np.zeros( (length, target_size**2) )
    for img_idx in range(length):

        # Get the image
        img = data[img_idx].reshape(original_size,original_size)

        # Resize the image
        img_resized = skimage.transform.resize(img, (target_size,target_size), anti_aliasing=True)

        # Put it back in vector form
        data_resized[img_idx] = img_resized.reshape(1, target_size**2)

    return data_resized

target_sizes = [12, 10, 8, 6]

for target_size in target_sizes[1:]:
    train_data_resized = resizeData(28, 12, train_data)
    test_data_resized = resizeData(28, 12, test_data)
    print('Successfully resized data to: ', target_size)

    model = RandomForest(train_data_resized, train_labels, test_labels, test_data_resized)
    results.append(model['accuracy'])

plt.figure()
plt.plot(results, target_sizes)
plt.title('Accuracy vs resolution of images')
plt.axis('off')
plt.show()