import os
import numpy as np
from joblib import dump, load
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks



# Load data from .pkl files. pkl file is demo_IMG_9453.pkl


# bad pushup
bad_data = load('/Users/adityatadimeti/gyML/demo_IMG_9456-2.pkl')

# start of rep: frame 150
# middle of rep: frame 175
# end of rep: frame 200

# end of exercises: frame 590

bad_start = bad_data['outputs//_DEMO/IMG_9456/img/000150.jpg']['pose'][0]
bad_middle = bad_data['outputs//_DEMO/IMG_9456/img/000175.jpg']['pose'][0]
bad_end = bad_data['outputs//_DEMO/IMG_9456/img/000200.jpg']['pose'][0]

print(bad_start.shape)
print(np.linalg.norm(bad_start-bad_middle), np.linalg.norm(bad_middle-bad_end), np.linalg.norm(bad_start-bad_end))

bad_similarity = []
for frame_num in range(151, 591):
    frame_pose = bad_data[f'outputs//_DEMO/IMG_9456/img/{frame_num:06d}.jpg']['pose'][0]
    bad_similarity.append(np.linalg.norm(bad_start - frame_pose))

# 'similarity' should be your array containing the raw pose similarity data.

# Apply a Gaussian filter for smoothing
# The 'sigma' parameter controls the amount of smoothing.
# A larger 'sigma' will result in a smoother signal.
sigma = 10  # This can be adjusted based on how much smoothing you need
bad_smoothed_similarity = gaussian_filter1d(bad_similarity, sigma=sigma)

# Assume 'similarity' is your array of pose similarity data
bad_peaks, _ = find_peaks(bad_smoothed_similarity)
bad_valleys, _ = find_peaks(-np.array(bad_smoothed_similarity))

#print(bad_peaks)
# Count the repetitions by counting the number of peaks or valleys

# bad_number_of_repetitions = max(len(bad_peaks), len(bad_valleys))
# print(f'Number of repetitions: {bad_number_of_repetitions}')

plt.plot(range(151, 591), bad_smoothed_similarity)
plt.xlabel('Frame Number')
plt.ylabel('L2 Norm')
plt.title('L2 Norm of Difference Vector Between Start Pose and Each Frame')
plt.show()


good_data = load('/Users/adityatadimeti/gyML/demo_IMG_9453.pkl')

# start of rep: frame 115
# end of entire exercise: frame 520

good_start = good_data['outputs//_DEMO/IMG_9453/img/000115.jpg']['pose'][0]
good_end = good_data['outputs//_DEMO/IMG_9453/img/000520.jpg']['pose'][0]

good_similarity = []
for frame_num in range(175, 520):
    frame_pose = good_data[f'outputs//_DEMO/IMG_9453/img/{frame_num:06d}.jpg']['pose'][0]
    good_similarity.append(np.linalg.norm(good_start - frame_pose))

# 'similarity' should be your array containing the raw pose similarity data.
# For demonstration purposes, I'm assuming it's a numpy array.

# Apply a Gaussian filter for smoothing
# The 'sigma' parameter controls the amount of smoothing.
# A larger 'sigma' will result in a smoother signal.
sigma = 10  # This can be adjusted based on how much smoothing you need
good_smoothed_similarity = gaussian_filter1d(good_similarity, sigma=sigma)

# Assume 'similarity' is your array of pose similarity data
good_peaks, _ = find_peaks(good_smoothed_similarity)
good_valleys, _ = find_peaks(-np.array(good_smoothed_similarity))

#print(bad_peaks)
# Count the repetitions by counting the number of peaks or valleys
good_number_of_repetitions = max(len(good_peaks), len(good_valleys))

for peak in good_peaks:
    print(f'Peak at frame {peak + 175}')

for valley in good_valleys:
    print(f'Valley at frame {valley + 175}')

plt.plot(range(175, 520), good_smoothed_similarity)
plt.xlabel('Frame Number')
plt.ylabel('Similarity')
plt.title('Similarity between Start Pose and Each Frame')
plt.show()
print(f'Number of repetitions: {good_number_of_repetitions}')

# Now print the average norm between the start and each peak and valley pose
# good_peak_norms = []
# for peak in good_peaks:
#     peak_pose = good_data[f'outputs//_DEMO/IMG_9453/img/{peak + 175:06d}.jpg']['pose'][0]
#     good_peak_norms.append(np.linalg.norm(good_start - peak_pose))
# print(f'Average norm between start and peak: {np.mean(good_peak_norms)}')

# good_valley_norms = []
# for valley in good_valleys:
#     valley_pose = good_data[f'outputs//_DEMO/IMG_9453/img/{valley + 175:06d}.jpg']['pose'][0]
#     good_valley_norms.append(np.linalg.norm(good_start - valley_pose))
# print(f'Average norm between start and valley: {np.mean(good_valley_norms)}')

# Now, compute the average norm between each bad peak with the start good peak. Do the same for valleys.

bad_peak_to_good_baseline = []
for bad_peak in bad_peaks:
    bad_peak_pose = bad_data[f'outputs//_DEMO/IMG_9456/img/{bad_peak + 151:06d}.jpg']['pose'][0]
    bad_peak_to_good_baseline.append(np.linalg.norm(good_start - bad_peak_pose))
print(f'Average norm between bad peak and start good pose: {np.mean(bad_peak_to_good_baseline)}')

bad_valley_to_good_baseline = []
for bad_valley in bad_valleys:
    bad_valley_pose = bad_data[f'outputs//_DEMO/IMG_9456/img/{bad_valley + 151:06d}.jpg']['pose'][0]
    bad_valley_to_good_baseline.append(np.linalg.norm(good_start - bad_valley_pose))
print(f'Average norm between bad valley and start good pose: {np.mean(bad_valley_to_good_baseline)}')
