import os
import csv
import torch
from PIL import Image
from jeans.inference import BagPredictor
import numpy as np

#set print to show 2 decpoint
torch.set_printoptions(precision=2, sci_mode=False)
np.set_printoptions(precision=2, suppress=True)

IMAGE_PATH = '/home/deepvisionpoc/Desktop/Jeans/SOLIDER_exp/SOLIDER-PersonAttributeRecognition/00a7023c-c80e-4459-9b46-ec9229634e16/2024-07-01T15-00-34_FLAG_P_FID_207_OID_63g0_ctw-cf-1c-030_1719820800_1719820834_207.jpg'
MODEL_CKPT = '/home/deepvisionpoc/Desktop/Jeans/SOLIDER_exp/SOLIDER-PersonAttributeRecognition/exp_result/BagCountOxygenLabeller/swin_s.bc_labeller/img_model/from_stracht_mon_sk.pth'

model = BagPredictor(model_ckpt=MODEL_CKPT)

# pred_class , pred_probs = model.run_inference(IMAGE_PATH)
# print(pred_class)
# print(pred_probs)

for root, dirs , files in os.walk('/home/deepvisionpoc/Desktop/Jeans/SOLIDER_exp/SOLIDER-PersonAttributeRecognition/mytestset/many_hand_bag'):
    for file in files:
        print("*"*60)
        
        full_p = os.path.join(root, file)
        print(full_p)
        pred_class , pred_probs = model.run_inference(full_p)
        print(pred_class)
        print(pred_probs)
        print("*"*60)