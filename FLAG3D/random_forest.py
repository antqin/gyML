import os
import numpy as np
from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier
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

def pad_and_flatten(array_list):
    # Determine the maximum length among all flattened arrays
    max_length = max(arr.size for arr in array_list)
    
    # Pad each array to match the maximum length and then flatten
    padded_list = []
    for arr in array_list:
        padded_arr = np.pad(arr, ((0, 0), (0, max_length - arr.size)))
        flattened_padded_arr = padded_arr.flatten()
        padded_list.append(flattened_padded_arr)
    
    return np.array(padded_list)

# Load train, dev, and test data
X_train, y_train = load_data_from_directory('dataset/train')
X_dev, y_dev = load_data_from_directory('dataset/dev')
X_test, y_test = load_data_from_directory('dataset/test')
X_train = pad_and_flatten(X_train)
X_dev = pad_and_flatten(X_dev)
X_test = pad_and_flatten(X_test)

# Load the trained model
# clf = load('random_forest_model.pkl')

# Initialize the Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42, verbose=1)

# Train the model
clf.fit(X_train, y_train)
dump(clf, 'random_forest_model.pkl')

# Predict on the dev set (or test set)
y_pred_dev = clf.predict(X_dev)

# Calculate the accuracy
accuracy = accuracy_score(y_dev, y_pred_dev)
print(f"Baseline Accuracy on Dev set: {accuracy:.2f}")

# If the dev set performance is satisfactory, you can evaluate on the test set
y_pred_test = clf.predict(X_test)
accuracy_test = accuracy_score(y_test, y_pred_test)
print(f"Accuracy on Test set: {accuracy_test:.2f}")

# Confusion Matrix for Random Forest
cm_rf = confusion_matrix(y_dev, y_pred_dev)
plt.figure(figsize=(12, 10))
sns.heatmap(cm_rf, annot=True, fmt='g', cmap='Blues')
plt.title('Confusion Matrix for Random Forest')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.savefig("confusion_matrix_rf.png", dpi=300)
plt.show()

# Feature Importance Visualization
plt.figure(figsize=(15, 5))
plt.bar(range(len(X_train[0])), clf.feature_importances_)
plt.title('Feature Importances for Random Forest')
plt.xlabel('Features')
plt.ylabel('Importance Value')
plt.savefig("feature_importances_rf.png", dpi=300)
plt.show()