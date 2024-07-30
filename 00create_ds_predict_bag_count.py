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


DATA_DATE = '2024-04-12'
CHUNK_NUM = 3
# DATASET_ROOT = f'/home/deepvisionpoc/Desktop/Jeans/resources/bag_count/raw/{DATA_DATE}_chunk_{CHUNK_NUM}/images'
DATASET_ROOT = '/home/deepvisionpoc/Desktop/Jeans/resources/mon'
DEST_IMG_ROOT = f'/home/deepvisionpoc/Desktop/Jeans/resources/bc_store/mon_preds_all_b'
CSV_FILE = f'/home/deepvisionpoc/Desktop/Jeans/resources/bc_store/mon_preds_all_b/label_result.csv'
PKL_FILE = f'/home/deepvisionpoc/Desktop/Jeans/resources/bc_store/mon_preds_all_b/label_result.pkl'
MODEL_CKPT = '/home/deepvisionpoc/Desktop/Jeans/SOLIDER_exp/SOLIDER-PersonAttributeRecognition/results/mon_songkran/MonAndSK_chunk0_0.pth'
# MODEL_CKPT ='/home/deepvisionpoc/Desktop/Jeans/SOLIDER_exp/SOLIDER-PersonAttributeRecognition/exp_result/ctw_store-match-bag_2024-07-01_labeled/swin_s.bc_ctw_store-match-bag_2024-07-01_labeled/img_model/ckpt_max_2024-07-26_17:57:31.pth'
os.makedirs(DEST_IMG_ROOT,exist_ok=True)


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
                    # pred_class = pred_class.item()
                    # print(pred_class)
                    # print(pred_probs)
                    # breakpoint()
                    
                    if pred_class <= 1:# and pred_probs[pred_class] >= 0.75:
                        print(f"Skipped {pred_class}  : {image_name}")
                        continue
                    
                    labeled_path = os.path.join(DEST_IMG_ROOT, str(pred_class))
                    os.makedirs(labeled_path, exist_ok=True)
                    shutil.copy(image_full_path, labeled_path)
                    # 0 is non-user-validated , 1 is user validated once
                    writer.writerow([image_name, pred_class, 0])
                    print(f"Class {pred_class} : {image_name} copied to destination")
                except:
                    raise Exception

predict_dataset_label()