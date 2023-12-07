import os
import numpy as np
from joblib import dump, load
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Load data from .pkl files
def load_data_from_directory(directory):
    X = []
    y = []
    for filename in os.listdir(directory):
        if filename.endswith(".pkl"):
            filepath = os.path.join(directory, filename)
            try:
                data = load(filepath)
                poses_data = data['poses']
                X.append(poses_data)  # Use only the 'poses' data as features
                action = filename.split('A')[1][:3]  # Extracting the Axxx part
                y.append(action)
            except Exception as e:
                print(f"Warning: Could not load {filename}. Error: {e}")
                continue
    return X, y

def get_max_len(array_list):
    return max(arr.shape[0] for arr in array_list)

# Function to pad arrays in a list to have the same length
def pad_arrays(array_list, len):
    # Pad each array to match the maximum length
    padded_list = []
    for arr in array_list:
        padded_arr = np.pad(arr, (0, len - arr.shape[0]))
        padded_list.append(padded_arr)
    
    return np.array(padded_list)

# Load train data
X_train, y_train = load_data_from_directory('dataset/train')
X_train = [x.flatten() for x in X_train]  # Flatten the 'poses' arrays for each training example

print(len(X_train[0]))
max_len = get_max_len(X_train)
X_train = pad_arrays(X_train, max_len)
print(len(X_train[0]))

# Apply PCA to training data
pca = PCA(n_components=0.75)
X_train_pca = pca.fit_transform(X_train)

# Plotting the explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_explained_variance = np.cumsum(explained_variance_ratio)

# Increase fontsize and bold text
plt.figure(figsize=(10, 6))

# Plot the explained variance ratio
plt.subplot(1, 2, 1)
plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio)
plt.title('Explained Variance Ratio', fontsize=16, fontweight='bold')
plt.xlabel('Principal Component', fontsize=14)
plt.ylabel('Variance Ratio', fontsize=14)

# Plot the cumulative explained variance
plt.subplot(1, 2, 2)
plt.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, marker='o')
plt.title('Cumulative Explained Variance', fontsize=16, fontweight='bold')
plt.xlabel('Number of Principal Components', fontsize=14)
plt.ylabel('Cumulative Variance Ratio', fontsize=14)

plt.tight_layout()
plt.show()