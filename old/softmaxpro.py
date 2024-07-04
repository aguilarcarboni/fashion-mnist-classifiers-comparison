import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import zipfile
import os

# Pre-process the raw data files and convert them into a format suitable for Scikit-Learn classifiers
def load_images(filepath):
    with open(filepath, 'rb') as f:
        f.read(16)
        data = np.frombuffer(f.read(), dtype=np.uint8)
        data = data.reshape(-1, 28*28)
    return data

def load_labels(filepath):
    with open(filepath, 'rb') as f:
        f.read(8)
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

data_folder = 'mnist_dataset/'
train_images_path = os.path.join(data_folder, 'train-images.idx3-ubyte')
train_labels_path = os.path.join(data_folder, 'train-labels.idx1-ubyte')
test_images_path = os.path.join(data_folder, 't10k-images.idx3-ubyte')
test_labels_path = os.path.join(data_folder, 't10k-labels.idx1-ubyte')

train_images = load_images(train_images_path)
train_labels = load_labels(train_labels_path)
test_images = load_images(test_images_path)
test_labels = load_labels(test_labels_path)

train_images = train_images / 255.0
test_images = test_images / 255.0

X = np.concatenate((train_images, test_images), axis=0)
y = np.concatenate((train_labels, test_labels), axis=0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Softmax Regression classifier
classifier = LogisticRegression(solver='lbfgs', max_iter=1000, multi_class='multinomial')
param_grid = {'C': [0.1, 1, 10]}
grid_search = GridSearchCV(classifier, param_grid, cv=3, scoring='accuracy')
grid_search.fit(X_train, y_train)

best_classifier = grid_search.best_estimator_
y_pred = best_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Softmax Regression Best Parameters: {grid_search.best_params_}")
print(f"Softmax Regression Test Accuracy: {accuracy}")
