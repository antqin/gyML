import os
import numpy as np
from joblib import dump, load
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, recall_score, precision_score
import matplotlib.pyplot as plt
import seaborn as sns

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


# Apply PCA to training data
pca = PCA(n_components=59)

# Load test data
X_test, y_test = load_data_from_directory('dataset/test')
X_test = [x.flatten() for x in X_test]  # Flatten the 'poses' arrays for each test example

max_len = 222408
X_test = pad_arrays(X_test, max_len)

X_test = pca.fit_transform(X_test)

# Load the trained model
clf = load('pca75_logreg_100iter.pkl')
print('Model Loading Complete...')
y_pred_test = clf.predict(X_test)

# Calculate accuracy, recall, and precision on the test set
accuracy_test = accuracy_score(y_test, y_pred_test)
recall_test = recall_score(y_test, y_pred_test, average='weighted')
precision_test = precision_score(y_test, y_pred_test, average='weighted')

# Print the results
print(f"Number of Features: {clf.coef_.shape[1]}")
print(f"Accuracy on Test set: {accuracy_test:.2f}")
print(f"Recall on Test set: {recall_test:.2f}")
print(f"Precision on Test set: {precision_test:.2f}")
