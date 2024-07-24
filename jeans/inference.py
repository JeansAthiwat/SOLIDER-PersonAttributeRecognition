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
from models.backbone.swin_transformer import swin_small_patch4_window7_224

from jeans.utils import get_wrong_pred, save_wrong_pred_figure
from PIL import Image

model_dict = {
    'swin_t': 768,
    'swin_s': 768,
    'swin_b': 1024,
}

class BagPredictor():
    def __init__(self, backbone_name='swin_s' , model_ckpt=None, attr_num=4,):
        
        assert torch.cuda.is_available(), "Cant build the model without cuda habibi"
        assert not model_ckpt == None, "dafuq bro no ckpt found"
        
        self.transform = get_transform(cfg)[1]
        
        if backbone_name.lower() == 'swin_s':
            self.backbone = swin_small_patch4_window7_224()
            
        self.c_output = model_dict[backbone_name]
    
        self.classifier = base_block.LinearClassifier(
            nattr=attr_num,
            c_in=self.c_output,
            bn=False,
            pool='avg',
            scale = 1
        )
        
        self.model = FeatClassifier(self.backbone, self.classifier, bn_wd=False)
        self.model = torch.nn.DataParallel(self.model).cuda()
        self.model = get_reload_weight("", self.model, pth=model_ckpt)
        self.model.eval()
        
        print("Bag Quantity Predictor: Initialized successfully")
        
    def run_inference(self,path_to_img):
        with torch.no_grad():
            img = self.preprocess(path_to_img).unsqueeze(0)
            valid_logits, _ = self.model(img) # logits and attention 
            # print(valid_logits)
            
            pred_probs = torch.sigmoid(valid_logits[0])
            pred_class = torch.argmax(pred_probs,1)
            # print(pred_probs)
            # print(pred_class)
            
            return pred_class , pred_probs
    
    def preprocess(self, img_path):
        img = Image.open(img_path)

        if self.transform is not None:
            img = self.transform(img)

        return img.cuda()