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

# Load data from .pkl files
def load_data_from_filename(pathname):
    X = []
    y = []
    try:
        data = load(pathname)
        poses_data = data['poses']
        X.append(poses_data)  # Use only the 'poses' data as features
        action = pathname.split('A')[1][:3]  # Extracting the Axxx part
        #print(action, poses_data)
        y.append(action)
    except Exception as e:
        print(f"Warning: Could not load {pathname}. Error: {e}")

    return X, y

def custom_pushup_process(pathname):
    X = []
    y = []
    data = load(pathname)
    # print(data['outputs//_DEMO/IMG_9453/img/000001.jpg']['pose'][0])
    print(len(data))
    #print(data['outputs//_DEMO/IMG_9453/img/000001.jpg']['pose'])
    # if ('outputs//_DEMO/IMG_9453/img/000001.jpg' in data):
    #     print(len(data['outputs//_DEMO/IMG_9453/img/000001.jpg']['pose'][0]))
    #     return
    # else:
    #     print(len(data['poses']))
    poses_data = data['poses']


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

import numpy as np

import numpy as np
from scipy.spatial.distance import cosine, cityblock
from scipy.stats import pearsonr
from fastdtw import fastdtw

def compute_similarity(file1_data, file2_data, metric='cosine'):
    # Assuming file1_data and file2_data are lists of numpy arrays
    # with each array representing pose data at a given time

    # Find the length of the shorter file
    min_length = min(len(file1_data), len(file2_data))

    # Calculate the end index for the chunk in the middle
    end_index = min_length // 2

    # Extract the chunk from each file
    chunk1 = file1_data[:end_index]
    chunk2 = file2_data[:end_index]

    # Compute similarity based on the specified metric
    if metric == 'euclidean':
        return np.linalg.norm(chunk1 - chunk2)
    elif metric == 'cosine':
        return cosine(np.hstack(chunk1), np.hstack(chunk2))
    elif metric == 'manhattan':
        return cityblock(np.hstack(chunk1), np.hstack(chunk2))
    elif metric == 'correlation':
        corr, _ = pearsonr(np.hstack(chunk1), np.hstack(chunk2))
        return corr
    elif metric == 'dtw':
        distance, path = fastdtw(chunk1, chunk2)
        return distance
    else:
        raise ValueError("Unknown metric specified")



# similarity_dict maps action to another dictionary. This sub-dictionary maps each other action to its similarity score with the action in the outer dictionary.

# Create list of all file names in the train, dev, and test directories.

def generate_similarity_dict(components_param):
    similarity_dict = {}
    train_dir = 'FLAG3D/dataset/train'
    dev_dir = 'FLAG3D/dataset/dev'
    test_dir = 'FLAG3D/dataset/test'

    train_files = [os.path.join(train_dir, file) for file in os.listdir(train_dir)]
    dev_files = [os.path.join(dev_dir, file) for file in os.listdir(dev_dir)]
    test_files = [os.path.join(test_dir, file) for file in os.listdir(test_dir)]

    all_files = train_files + dev_files + test_files

    # Create a dictionary that maps an action to a list of filepaths for that action
    action_to_filepaths = {}
    for file in tqdm(all_files):
        # Extract action from filename, removing the FLAG3D/dataset part, as well as the train, dev, or test parts
        action = file.split('FLAG3D/dataset/')[1].split('/')[1].split('A')[1][:3]
        if action not in action_to_filepaths:
            action_to_filepaths[action] = []
        action_to_filepaths[action].append(file)
    
    for action in tqdm(action_to_filepaths):
        # Select random file from list of filepaths for that action
        file = np.random.choice(action_to_filepaths[action])
        similarity_dict[action] = {}

        # Load data from file
        file_data = load_data_from_filename(file)


        # Compute similarity score between this action and all other actions
        for other_action in action_to_filepaths:
            # Select random file from list of filepaths for that action
            other_file = np.random.choice(action_to_filepaths[other_action])

            # Load data from other file
            other_file_data = load_data_from_filename(other_file)

            # components = 0.75 * min(file_data[0][0].shape[1], other_file_data[0][0].shape[1])

            components = components_param
            pca = PCA(n_components = components)

            pca_data = pca.fit_transform(file_data[0][0])
            other_file_pca_data = pca.transform(other_file_data[0][0])

            # Compute similarity score between file and other file
            #similarity_score = compute_similarity(file_data[0][0], other_file_data[0][0])

            pca_similarity_score = compute_similarity(pca_data, other_file_pca_data)


            # Add entry to similarity dictionary
            #similarity_dict[action][other_action] = similarity_score
            similarity_dict[action][other_action] = pca_similarity_score
            #similarity_dict[other_action][action] = similarity_score

    return similarity_dict

# For each action, print the lowest similarity score with any other action as well as the action it is most similar to
def print_similarity_dict(similarity_dict):
    num_actions = len(similarity_dict)
    num_same_action = 0
    
    for action in similarity_dict:
        min_similarity = float('inf')
        min_action = None
        for other_action in similarity_dict[action]:
            if similarity_dict[action][other_action] < min_similarity:
                min_similarity = similarity_dict[action][other_action]
                min_action = other_action
        
        if min_action == action:
            num_same_action += 1
    
    percentage_same_action = (num_same_action / num_actions) * 100
    print("Percentage of actions whose most similar action is itself:", percentage_same_action)
    return percentage_same_action
#print_similarity_dict(generate_similarity_dict())

# components_arr = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# metrics = ['euclidean', 'cosine', 'manhattan', 'correlation', 'dtw']

components_arr = [0.9, 0.95, 0.975, 0.999]
metrics = ['manhattan', 'dtw']

for metric in tqdm(metrics):
    for components in tqdm(components_arr):
        similarity_dict = generate_similarity_dict(components)

        print("Components:", components, "Metric:", metric)
        print_similarity_dict(similarity_dict)
        print()
# Example usage
# Assuming pushup_train_data_1 and pushup_train_data_3 are your pose data arrays
#similarity_score = compute_similarity(pushup_train_data_1, pushup_train_data_3)

# now compute similarity score for pushup_train_data_1 and array of 0 values of same dimension as pushup_train_data_3
# similarity_score = compute_similarity(pushup_train_data_1, random_train_data_3)
# print("Similarity Score:", similarity_score)
