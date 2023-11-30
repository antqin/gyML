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

# Load train, dev, and test data
X_train, y_train = load_data_from_directory('dataset/train')
X_train = [x.flatten() for x in X_train]  # Flatten the 'poses' arrays for each training example

X_dev, y_dev = load_data_from_directory('dataset/dev')
X_dev = [x.flatten() for x in X_dev]  # Flatten the 'poses' arrays for each dev example

X_test, y_test = load_data_from_directory('dataset/test')
X_test = [x.flatten() for x in X_test]  # Flatten the 'poses' arrays for each test example

print(len(X_train[0]))
print(len(X_dev[0]))
print(len(X_test[0]))
train_dimension = get_max_len(X_train)
dev_dimension = get_max_len(X_dev)
test_dimension = get_max_len(X_test)
max_len = max(max(train_dimension, dev_dimension), test_dimension)
X_train = pad_arrays(X_train, max_len)
X_dev = pad_arrays(X_dev, max_len)
X_test = pad_arrays(X_test, max_len)
print(len(X_train[0]))
print(len(X_dev[0]))
print(len(X_test[0]))

# Load the trained model
# clf = load('logreg_model_100.pkl')

# Initialize the Logistic Regression model with multinomial strategy
clf = LogisticRegression(max_iter=100, multi_class='multinomial', solver='saga', verbose=1)

# Apply PCA to training data
pca = PCA(n_components=0.75)
X_train_pca = pca.fit_transform(X_train)

# Train the model
clf.fit(X_train_pca, y_train)
dump(clf, 'pca75_logreg_100iter.pkl')

# Predict on the dev set (or test set)
X_dev_pca = pca.transform(X_dev)
y_pred_dev = clf.predict(X_dev_pca)

# Calculate the accuracy
accuracy = accuracy_score(y_dev, y_pred_dev)
print(f"Baseline Accuracy on Dev set: {accuracy:.2f}")

# If the dev set performance is satisfactory, you can evaluate on the test set
X_test_pca = pca.transform(X_test)
y_pred_test = clf.predict(X_test_pca)
accuracy_test = accuracy_score(y_test, y_pred_test)
print(f"Accuracy on Test set: {accuracy_test:.2f}")

# Confusion Matrix for Logistic Regression
cm_logistic = confusion_matrix(y_dev, y_pred_dev)
plt.figure(figsize=(12, 10))
sns.heatmap(cm_logistic, annot=True, fmt='g', cmap='Blues')
plt.title('Confusion Matrix for Logistic Regression')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.savefig("pca_75_logreg_mc_confusion_matrix.png", dpi=300)

# Coefficients Visualization
# plt.figure(figsize=(15, 5))
# plt.bar(range(len(X_train[0])), clf.coef_[0])  # Change index for other classes
# plt.title('Feature Coefficients for Logistic Regression (Class 0)')
# plt.xlabel('Features')
# plt.ylabel('Coefficient Value')
# plt.savefig("logreg_mc_feature_coefficients.png", dpi=300)
# plt.show()
