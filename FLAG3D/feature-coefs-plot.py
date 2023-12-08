import os
from joblib import load
import matplotlib.pyplot as plt

# Load the trained model
clf = load('pca75_logreg_100iter.pkl')
print('Model Loading Complete...')


# Coefficients Visualization
plt.figure(figsize=(6, 6))
plt.bar(range(59), clf.coef_[0])  # Change index for other classes
plt.title('Feature Coefficients for Logistic Regression (Class 0)')
plt.xlabel('Features')
plt.ylabel('Coefficient Value')
plt.savefig("temp.png", dpi=300)
plt.tight_layout()
plt.show()
