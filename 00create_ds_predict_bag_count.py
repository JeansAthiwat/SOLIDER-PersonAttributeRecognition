import argparse
import json
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import pickle

from dataset.augmentation import get_transform
# from dataset.multi_label.coco import COCO14
from metrics.pedestrian_metrics import get_pedestrian_metrics
from models.model_factory import build_backbone, build_classifier

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs import cfg, update_config
from dataset.pedes_attr.pedes import PedesAttr
from metrics.ml_metrics import get_map_metrics, get_multilabel_metrics
from models.base_block import FeatClassifier
# from models.model_factory import model_dict, classifier_dict

from tools.function import get_model_log_path, get_reload_weight
from tools.utils import set_seed, str2bool, time_str


import argparse
import pickle
import numpy as np

from configs import cfg, update_config
#from dataset.multi_label.coco import COCO14
from dataset.augmentation import get_transform
from metrics.ml_metrics import get_multilabel_metrics
from metrics.pedestrian_metrics import get_pedestrian_metrics
from tools.distributed import distribute_bn
from tools.vis import tb_visualizer_pedes
import torch
from torch.utils.data import DataLoader

from dataset.pedes_attr.pedes import PedesAttr
from models.base_block import FeatClassifier
from models.model_factory import build_loss, build_classifier, build_backbone

from tools.function import get_model_log_path, get_reload_weight, seperate_weight_decay
from tools.utils import time_str, save_ckpt, ReDirectSTD, set_seed, str2bool, gen_code_archive
from models.backbone import swin_transformer
from losses import bceloss, scaledbceloss
from models import base_block

from jeans.utils import get_wrong_pred, save_wrong_pred_figure
from jeans.inference import BagPredictor
import shutil
import csv
import os
import shutil
import cv2
from easydict import EasyDict as edict

set_seed(605)
#set print to show 2 decpoint
torch.set_printoptions(precision=2, sci_mode=False)
np.set_printoptions(precision=2, suppress=True)

'''
esta function predict the amount of bags on all images in the chunk and save the picture in a new destination sorted by classes
'''
DATA_DATE = '2024-04-12'
CHUNK_NUM = 1
DATASET_ROOT = f'/home/deepvisionpoc/Desktop/Jeans/resources/bag_count/raw/{DATA_DATE}_chunk_{CHUNK_NUM}/images'
DEST_IMG_ROOT = f'/home/deepvisionpoc/Desktop/Jeans/resources/bag_count/labeled/{DATA_DATE}_chunk_{CHUNK_NUM}'
CSV_FILE = f'/home/deepvisionpoc/Desktop/Jeans/resources/bag_count/labeled/{DATA_DATE}_chunk_{CHUNK_NUM}/label_result.csv'
PKL_FILE = f'/home/deepvisionpoc/Desktop/Jeans/resources/bag_count/labeled/{DATA_DATE}_chunk_{CHUNK_NUM}/label_result.pkl'
MODEL_CKPT = '/home/deepvisionpoc/Desktop/Jeans/SOLIDER_exp/SOLIDER-PersonAttributeRecognition/results/mon_songkran/MonAndSK_chunk0_0.pth'

os.makedirs(DEST_IMG_ROOT,exist_ok=True)

#re label
LABELED_CSV = os.path.join(DEST_IMG_ROOT, 'label_result.csv')

CURRENT_CLASS = '0'

def predict_dataset_label():

    with open(CSV_FILE, 'w', newline='') as csvfile:
        pass  # This will create an empty file
    
    with open(CSV_FILE, 'a', newline='') as csv_f:  # 'a' mode for appending
        writer = csv.writer(csv_f)
        
        model = BagPredictor(model_ckpt=MODEL_CKPT)
        print(f"dataset_root : {DATASET_ROOT}")
        print(f"Destination_root : {DEST_IMG_ROOT}")


        for root , dirs , files in os.walk(DATASET_ROOT):
            for file in files:
                
                if not file.endswith(('.png','.jpg')):
                    continue
                
                image_full_path = os.path.join(root, file)
                image_name = os.path.basename(file)
                try:
                    pred_class , pred_probs = model.run_inference(image_full_path)
                    
                    # breakpoint()
                    labeled_path = os.path.join(DEST_IMG_ROOT, str(pred_class.item()))
                    os.makedirs(labeled_path, exist_ok=True)
                    shutil.copy(image_full_path, labeled_path)
                    # 0 is non-user-validated , 1 is user validated once
                    writer.writerow([image_name, pred_class.item(), 0])
                    print(f"Class {pred_class.item()} : {image_name} copied to destination")
                except:
                    raise Exception
                
    # predict_dataset_label()
    csv_dict = edict()
    with open(CSV_FILE, 'r', newline='') as csv_f:
        reader = csv.reader(csv_f)
        for row in reader:
            if row:
                img_name, label, state = row
                csv_dict[img_name] = edict()
                csv_dict[img_name]['label'] = label
                csv_dict[img_name]['state'] = state
        # breakpoint()

    with open(PKL_FILE, 'wb') as f:
        pickle.dump(csv_dict,f)
        
def update_dataset(dataset, image_name, label):
    #  csv_dict = edict()
                # csv_dict[img_name] = edict()
                # csv_dict[img_name]['label'] = label
                # csv_dict[img_name]['state'] = state
    dataset[image_name].label = label
    dataset[image_name].state = 1
    
    
def hand_validate_predicted_label():
    # Load labeled images from CSV
    with open(PKL_FILE, 'rb', newline='') as f:
        dataset = pickle.load(f)
                    
    # Process images
    for root, dirs, files in os.walk(os.path.join(DEST_IMG_ROOT, CURRENT_CLASS)):
        for file in files:
            if not file.endswith(('.jpg', 'png')):
                continue
            
            img_path = os.path.join(root, file)
            image_name = os.path.basename(img_path)
            
            if image_name not in dataset.keys():
                print("something went wrong")
                breakpoint()
                
            if (not dataset[image_name].state == 0):
                continue
                
            
            # Open image using OpenCV
            image = cv2.imread(img_path)
            
            # Resize image to fill the top and bottom
            screen_height = 1080  # Assuming a common screen height if window not visible
            screen_width = 1920  # Assuming a common screen width if window not visible
            aspect_ratio = image.shape[1] / image.shape[0]  # width / height
            
            new_height = screen_height
            new_width = int(new_height * aspect_ratio)
            
            if new_width > screen_width:
                new_width = screen_width
                new_height = int(new_width / aspect_ratio)
            
            resized_image = cv2.resize(image, (new_width, new_height))

            # Create a named window and set it to fullscreen
            cv2.namedWindow('Image', cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty('Image', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow('Image', resized_image)
            
            key = cv2.waitKey(0)
            if key == ord('d'):  # Delete
                update_dataset(dataset,image_name, 'delete')
                print(f"MARKED AS TO DELETE: {image_name}")
            elif key == ord('k'):  # Keep
                update_dataset(dataset,image_name, CURRENT_CLASS)
                print(f"OK: {image_name}")
            elif key == ord('0'):  # Move to class 0
                new_path = os.path.join(DEST_IMG_ROOT, '0', image_name)
                if not os.path.exists(os.path.join(DEST_IMG_ROOT, '0')):
                    os.makedirs(os.path.join(DEST_IMG_ROOT, '0'))
                shutil.move(img_path, new_path)
                update_dataset(dataset,image_name, '0')
                print(f"MOVED TO CLASS 0: {image_name}")
            elif key == ord('1'):  # Move to class 1
                new_path = os.path.join(DEST_IMG_ROOT, '1', image_name)
                if not os.path.exists(os.path.join(DEST_IMG_ROOT, '1')):
                    os.makedirs(os.path.join(DEST_IMG_ROOT, '1'))
                shutil.move(img_path, new_path)
                update_dataset(dataset,image_name, '1')
                print(f"MOVED TO CLASS 1: {image_name}")
            elif key == ord('2'):  # Move to class 2
                new_path = os.path.join(DEST_IMG_ROOT, '2', image_name)
                if not os.path.exists(os.path.join(DEST_IMG_ROOT, '2')):
                    os.makedirs(os.path.join(DEST_IMG_ROOT, '2'))
                shutil.move(img_path, new_path)
                update_dataset(dataset,image_name, '2')
                print(f"MOVED TO CLASS 2: {image_name}")
            elif key == ord('3'):  # Move to class 3
                new_path = os.path.join(DEST_IMG_ROOT, '3', image_name)
                if not os.path.exists(os.path.join(DEST_IMG_ROOT, '3')):
                    os.makedirs(os.path.join(DEST_IMG_ROOT, '3'))
                shutil.move(img_path, new_path)
                update_dataset(dataset,image_name, '3')
                print(f"MOVED TO CLASS 3: {image_name}")
            
            cv2.destroyAllWindows()
