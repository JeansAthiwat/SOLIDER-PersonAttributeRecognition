import os
from os import path as osp
import numpy as np
import glob
import shutil
from easydict import EasyDict as edict
import pickle
from sklearn.model_selection import train_test_split

DS_NAME = "SongkhranChunk0-0"

DATASET_ROOT = "/home/deepvisionpoc/Desktop/Jeans/resources/mon/2024-04-12_chunk_0"
DESTINATION_ROOT = f"/home/deepvisionpoc/Desktop/Jeans/SOLIDER_exp/SOLIDER-PersonAttributeRecognition/data/{DS_NAME}/images"
FORMATTED_RESPONSE = f"/home/deepvisionpoc/Desktop/Jeans/SOLIDER_exp/SOLIDER-PersonAttributeRecognition/data/{DS_NAME}/intermediate.pkl"
DATASET_ALL_PKL_PATH = f"/home/deepvisionpoc/Desktop/Jeans/SOLIDER_exp/SOLIDER-PersonAttributeRecognition/data/{DS_NAME}/dataset_all.pkl"

one_hot_matrix = np.eye(4, dtype=np.int64)

def flatten_dataset():
    os.makedirs(DESTINATION_ROOT, exist_ok=True)

    format_dict = edict()
    format_dict.description = "BagCount_{DS_NAME}"
    format_dict.reorder = "group_order"
    format_dict.root = f"/mnt/data1/jiajian/datasets/attribute/{DS_NAME}/images"
    format_dict.attr_name = ["0 Bags", "1 Bags", "2 Bags", ">= 3 Bags"]
    format_dict.label_idx = edict({"eval": [0, 1, 2, 3]})


    format_dict.partition = edict()
    format_dict.image_name = []  # list of images file name
    format_dict.label = []  # must be np.array()

    count = 0
    for root, dirs, files in os.walk(DATASET_ROOT):
        for file in files:
            image_full_path = osp.join(root, file)
            
            # if not "2024-04-12" in image_full_path:
            #     continue
            
            image_name = osp.basename(file)
            label = image_full_path.split("/")[-2]

            try:
                shutil.copy(image_full_path, DESTINATION_ROOT)
                format_dict.image_name.append(image_name)
                format_dict.label.append(min(3, int(label)))
                count += 1
            except Exception as e:
                raise e

    print("total_of : ", count)
    
    #post process
    format_dict.label = np.array(format_dict.label)
    
    with open(FORMATTED_RESPONSE, "wb") as f:
        pickle.dump(format_dict, f)
        
def create_partitions():
    with open(FORMATTED_RESPONSE, "rb") as f: 
        data = pickle.load(f)
        
    # First, split into train and temp (test + val)
    train_labels, temp_labels, train_indices, temp_indices = train_test_split(
        data.label,
        np.arange(len(data.label)),
        test_size=0.3,
        stratify=data.label,
        random_state=42,
    )
    
    # Then, split temp into test and val
    _, _, test_indices, val_indices = train_test_split(
        temp_labels, temp_indices, test_size=0.50, stratify=temp_labels, random_state=42
    )
    
    train_indices, val_indices, test_indices
    
    data.partition.train = np.array(train_indices)
    data.partition.val = np.array(val_indices)
    data.partition.test = np.array(test_indices)
    data.partition.trainval = np.array(np.concatenate((train_indices, val_indices)))
    print(np.unique(data.label, return_counts=True))
    data.label = one_hot_matrix[data.label]
    
    with open(DATASET_ALL_PKL_PATH, "wb") as f: 
        pickle.dump(data, f)
#run
flatten_dataset()
create_partitions()



