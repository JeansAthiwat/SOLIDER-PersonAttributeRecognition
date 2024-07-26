import os
from os import path as osp
import numpy as np
import glob
import shutil
from easydict import EasyDict as edict
import pickle
from sklearn.model_selection import train_test_split

DS_NAME = "TEST_SET_NO_BP_SB"

DATASET_ROOT = "/home/deepvisionpoc/Desktop/Jeans/resources/bag_count/test_set_clean/no_bp_sb"
DESTINATION_ROOT = f"/home/deepvisionpoc/Desktop/Jeans/SOLIDER_exp/SOLIDER-PersonAttributeRecognition/data/{DS_NAME}/images"
FORMATTED_RESPONSE = f"/home/deepvisionpoc/Desktop/Jeans/SOLIDER_exp/SOLIDER-PersonAttributeRecognition/data/{DS_NAME}/intermediate.pkl"
DATASET_ALL_PKL_PATH = f"/home/deepvisionpoc/Desktop/Jeans/SOLIDER_exp/SOLIDER-PersonAttributeRecognition/data/{DS_NAME}/dataset_all.pkl"

one_hot_matrix = np.eye(4, dtype=np.int64)

def flatten_dataset():
    os.makedirs(DESTINATION_ROOT, exist_ok=True)

    format_dict = edict()
    format_dict.description = f"BagCount_{DS_NAME}"
    format_dict.reorder = "group_order"
    format_dict.root = f"/mnt/data1/jiajian/datasets/attribute/{DS_NAME}/images"
    format_dict.attr_name = ["0 Bags", "1 Bags", "2 Bags", ">= 3 Bags"]
    format_dict.label_idx = edict({"eval": [0, 1, 2, 3]})


    format_dict.partition = edict()
    format_dict.image_name = []  # list of images file name
    format_dict.label = []  # must be np.array()
    # print(DATASET_ROOT)
    count = 0
    for root, dirs, files in os.walk(DATASET_ROOT):
        for file in files:
            
            if not file.endswith(('.jpg','.png')):
                continue
            
            image_full_path = osp.join(root, file)
            print(image_full_path)
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
    indices = np.arange(len(data.label))
    data.partition.train = indices
    data.partition.val = indices
    data.partition.test = indices
    data.partition.trainval = indices
    print(np.unique(data.label, return_counts=True))
    data.label = one_hot_matrix[data.label]
    
    with open(DATASET_ALL_PKL_PATH, "wb") as f: 
        pickle.dump(data, f)
#run

flatten_dataset()
create_partitions()



