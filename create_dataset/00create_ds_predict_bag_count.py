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

set_seed(605)


'''
esta function predict the amount of bags on all images in the chunk and save the picture in a new destination sorted by classes
'''
DATA_DATE = '2024-04-12'
CHUNK_NUM = 1
DATASET_ROOT = f'/home/deepvisionpoc/Desktop/Jeans/resources/bag_count/raw/{DATA_DATE}_chunk_{CHUNK_NUM}'
DEST_IMG_ROOT = f'/home/deepvisionpoc/Desktop/Jeans/resources/bag_count/labeled/{DATA_DATE}_chunk_{CHUNK_NUM}'
MODEL_CKPT = '/home/deepvisionpoc/Desktop/Jeans/SOLIDER_exp/SOLIDER-PersonAttributeRecognition/results/mon_songkran/MonAndSK_chunk0_0.pth'



# with open(LABELED_CSV, 'a', newline='') as csv_f:  # 'a' mode for appending
#     writer = csv.writer(csv_f)
#     writer.writerow([image_name, status])
        
model = BagPredictor(model_ckpt=MODEL_CKPT)
print(f"dataset_root : {DATASET_ROOT}")
print(f"Destination_root : {DEST_IMG_ROOT}")

#set print to show 2 decpoint
torch.set_printoptions(precision=2, sci_mode=False)
np.set_printoptions(precision=2, suppress=True)

for root , dirs , files in os.walk(DATASET_ROOT):
    for file in files:
        
        if not file.endswith(('.png','.jpg')):
            continue
        
        image_full_path = os.path.join(root, file)
        try:
            pred_class , pred_probs = model.run_inference(image_full_path)
        except:
            raise Exception
        
        labeled_path = os.path.join(DEST_IMG_ROOT, str(pred_class.item()))
        os.makedirs(labeled_path, exist_ok=True)
        shutil.copy(image_full_path, labeled_path)
        
        print(f"Class {pred_class} : {image_full_path} copied to destination")
