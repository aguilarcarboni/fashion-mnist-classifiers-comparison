# Project 1
# Jose Miguel Loguirato, Andres Aguilar, Maria Elena Leizaola
# CS 3368

import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

from mnist_helper import get_mnist_data_and_labels, resize_data

from naive_bayes import NaiveBayes
from ridge import RidgeRegression
from softmax import SoftMaxRegression
from random_forest import RandomForest

"""
Step 1: Pre-process the raw data files and convert 
them into a format suitable for Scikit-Learn classifiers
"""

# Load the training and test data from the Pickle file (or from other files if Pickle file does not exist)
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

# Print size information from the training dataset
print("Preprocessing complete.")
print("Training dataset size: ", train_data.shape)
print("Test dataset size: ", test_data.shape)

"""
Step 2: Explore the dataset (quantity of examples of each class, 
distribution of pixel values, centroid images: overall and per-class)
"""

# Plot quantity of examples of each class
hist, bins = np.histogram(test_labels, bins=10)
center = (bins[:-1] + bins[1:]) / 2
width = np.diff(bins)
plt.bar(center, hist, align='center', width=width, color=['red', 'blue'])
plt.show()

# Plot distribution of pixel values
hist, bins = np.histogram(train_data, bins=50)
center = (bins[:-1] + bins[1:]) / 2
width = np.diff(bins)
plt.bar(center, hist, align='center', width=width, color='red')
plt.show()

# Compute and plot overall centroid image
centroid_image = np.mean(train_data, axis=0)
plt.figure()
plt.imshow(centroid_image.reshape(28, 28), cmap='gray')
plt.title('Overall Centroid Image')
plt.axis('off')
plt.show()

# Compute and plot one centroid image per class
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

model = NaiveBayes(train_data, train_labels, test_labels, test_data)
if model['accuracy'] > best_model['accuracy']:
    best_model = model

model = RidgeRegression(train_data, train_labels, test_labels, test_data)
if model['accuracy'] > best_model['accuracy']:
    best_model = model

model = SoftMaxRegression(train_data, train_labels, test_labels, test_data)
if model['accuracy'] > best_model['accuracy']:
    best_model = model
    
results = []
model = RandomForest(train_data, train_labels, test_labels, test_data)
if model['accuracy'] > best_model['accuracy']:
    best_model = model

print('Best model:', best_model)
results.append(best_model['accuracy'])


"""
Step 5: Display several examples of images that were mis-classified by the best classifier.
"""

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


"""
Step 6: Study the pixel importances from the best classifier.  Display them as an image.
"""

coef_img = best_model['model'].feature_importances_.reshape(28, 28)
plt.figure()
plt.title('Feature importances as an image.')
plt.imshow(coef_img, cmap="gray_r")
plt.show()


"""
Step 7: 7.	Study the effect of image resolution on the accuracy of your best classifier.
"""
target_sizes = [12, 10, 8, 6]

for target_size in target_sizes[1:]:
    train_data_resized = resize_data(28, 12, train_data)
    test_data_resized = resize_data(28, 12, test_data)
    print(f'Successfully resized data to: {target_size}x{target_size}')

    model = RandomForest(train_data_resized, train_labels, test_labels, test_data_resized)
    results.append(model['accuracy'])

plt.figure()
plt.plot(target_sizes, results)
plt.title('Accuracy vs resolution of images')
plt.axis('off')
plt.show()