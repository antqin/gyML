import os
import random
from collections import defaultdict

# Path to the folder containing .pkl files
DATA_FOLDER = "subset_smpl_param"
TRAIN_FOLDER = "train"
DEV_FOLDER = "dev"
TEST_FOLDER = "test"

# Proportions for train, dev, and test splits
TRAIN_RATIO = 0.7
DEV_RATIO = 0.2
TEST_RATIO = 0.1

# Ensure reproducibility
SEED = 42
random.seed(SEED)

def split_dataset():
    # Create directories for train, dev, and test
    if not os.path.exists(TRAIN_FOLDER):
        os.makedirs(TRAIN_FOLDER)
    if not os.path.exists(DEV_FOLDER):
        os.makedirs(DEV_FOLDER)
    if not os.path.exists(TEST_FOLDER):
        os.makedirs(TEST_FOLDER)

    # List all .pkl files
    all_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith('.pkl')]
    
    # Create a dictionary to store files for each action
    action_dict = defaultdict(list)
    for file in all_files:
        # Extract action "Axxx" from filename
        action = file.split('P')[1].split('A')[0]
        action_dict[action].append(file)

    # Stratified sampling
    for action, files in action_dict.items():
        random.shuffle(files)

        train_count = int(len(files) * TRAIN_RATIO)
        dev_count = int(len(files) * DEV_RATIO)
        
        train_files = files[:train_count]
        dev_files = files[train_count:train_count+dev_count]
        test_files = files[train_count+dev_count:]
        
        # Move files to respective directories
        for file in train_files:
            os.rename(os.path.join(DATA_FOLDER, file), os.path.join(TRAIN_FOLDER, file))
        for file in dev_files:
            os.rename(os.path.join(DATA_FOLDER, file), os.path.join(DEV_FOLDER, file))
        for file in test_files:
            os.rename(os.path.join(DATA_FOLDER, file), os.path.join(TEST_FOLDER, file))

if __name__ == "__main__":
    split_dataset()
