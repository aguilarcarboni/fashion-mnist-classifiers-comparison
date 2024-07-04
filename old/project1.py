#project 1 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import zipfile
import os

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

##########################################################################################################
## --->> FIRST PART:
## 1. Pre-process the raw data files and convert them into a format suitable for Scikit-Learn classifiers

# Function to load images from the .ubyte file
def load_images(filepath):
    with open(filepath, 'rb') as f:
        f.read(16)  # Skip magic number and dimensions
        data = np.frombuffer(f.read(), dtype=np.uint8)
        data = data.reshape(-1, 28*28)
    return data

# Function to load labels from the .ubyte file
def load_labels(filepath):
    with open(filepath, 'rb') as f:
        f.read(8)  # Skip magic number and number of items
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

# Paths to your .ubyte files
data_folder = 'mnist_dataset/'
train_images_path = os.path.join(data_folder, 'train-images.idx3-ubyte')
train_labels_path = os.path.join(data_folder, 'train-labels.idx1-ubyte')
test_images_path = os.path.join(data_folder, 't10k-images.idx3-ubyte')
test_labels_path = os.path.join(data_folder, 't10k-labels.idx1-ubyte')

# Load the dataset
train_images = load_images(train_images_path)
train_labels = load_labels(train_labels_path)

test_images = load_images(test_images_path)
test_labels = load_labels(test_labels_path)

# Normalize the images
train_images = train_images / 255.0
test_images = test_images / 255.0

# Combine the data for easier handling
X = np.concatenate((train_images, test_images), axis=0)
y = np.concatenate((train_labels, test_labels), axis=0)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to DataFrame
train_df = pd.DataFrame(X_train)
train_df['label'] = y_train

test_df = pd.DataFrame(X_test)
test_df['label'] = y_test

# Save to CSV
train_csv_path = 'fashion_mnist_train.csv'
test_csv_path = 'fashion_mnist_test.csv'
train_df.to_csv(train_csv_path, index=False)
test_df.to_csv(test_csv_path, index=False)

# Create a zip archive
archive_name = 'fashion_mnist_dataset.zip'
with zipfile.ZipFile(archive_name, 'w') as archive:
    archive.write(train_csv_path)
    archive.write(test_csv_path)

# Clean up the CSV files if needed
os.remove(train_csv_path)
os.remove(test_csv_path)

print(f"Preprocessing complete. Data saved to archive '{archive_name}'.")

##########################################################################################################
## --->> SECOND PART:
## 2. Explore the dataset: quantity of examples of each class, 
##    distribution of pixel values, centroid images: overall and per-class

# Quantity of examples of each class
class_counts = train_df['label'].value_counts().sort_index()

plt.figure(figsize=(10, 6))
sns.barplot(x=class_counts.index, y=class_counts.values, hue=class_counts.index, palette="viridis", dodge=False)
plt.xlabel('Class')
plt.ylabel('Number of examples')
plt.title('Quantity of examples of each class in the training set')
plt.legend([], [], frameon=False)  # Remove the legend
plt.show()

# Distribution of pixel values
plt.figure(figsize=(10, 6))
sns.histplot(X_train.flatten(), bins=50, kde=True, color='purple')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.title('Distribution of pixel values in the training set')
plt.show()

# Centroid images: overall and per-class
centroid_image = np.mean(X_train, axis=0)
centroid_image = centroid_image.reshape(28, 28)

plt.figure(figsize=(6, 6))
plt.imshow(centroid_image, cmap='gray')
plt.title('Overall Centroid Image')
plt.axis('off')
plt.show()

fig, axes = plt.subplots(2, 5, figsize=(15, 6))
fig.suptitle('Centroid Images by Class')

for i in range(10):
    class_centroid_image = np.mean(X_train[y_train == i], axis=0)
    class_centroid_image = class_centroid_image.reshape(28, 28)
    ax = axes[i // 5, i % 5]
    ax.imshow(class_centroid_image, cmap='gray')
    ax.set_title(f'Class {i}')
    ax.axis('off')

plt.tight_layout()
plt.show()


##########################################################################################################
## --->> THIRD PART:
## 3. Attempt multiple classifiers (at least 4) and optimize their parameters

# Initialize classifiers
classifiers = {
    'Naive Bayes': GaussianNB(),
    'Softmax Regression': LogisticRegression(solver='lbfgs', max_iter=1000),
    'Random Forest': RandomForestClassifier(),
    'kNN': KNeighborsClassifier()
}

# Define parameter grids for each classifier
param_grids = {
    'Naive Bayes': {},
    'Softmax Regression': {'C': [0.1, 1, 10]},
    'Random Forest': {'n_estimators': [100, 200], 'max_features': ['sqrt', 'log2']},
    'kNN': {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']}
}

best_estimators = {}
best_scores = {}

# Perform grid search for each classifier
for name, clf in classifiers.items():
    grid_search = GridSearchCV(clf, param_grids[name], cv=3, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    best_estimators[name] = grid_search.best_estimator_
    best_scores[name] = grid_search.best_score_
    print(f"{name} Best Parameters: {grid_search.best_params_}")
    print(f"{name} Best CV Score: {grid_search.best_score_}")

# Evaluate the best models on the test set
test_scores = {}
for name, estimator in best_estimators.items():
    y_pred = estimator.predict(X_test)
    test_scores[name] = accuracy_score(y_test, y_pred)
    print(f"{name} Test Accuracy: {test_scores[name]}")

# Plot the results
plt.figure(figsize=(10, 6))
sns.barplot(x=list(test_scores.keys()), y=list(test_scores.values()), palette="viridis")
plt.xlabel('Classifier')
plt.ylabel('Test Accuracy')
plt.title('Test Accuracy of Different Classifiers')
plt.ylim(0, 1)
plt.show()
