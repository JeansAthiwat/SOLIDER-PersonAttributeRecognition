import os
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import csv
import torch
from PIL import Image
from jeans.inference import BagPredictor

#set print to show 2 decpoint
torch.set_printoptions(precision=2, sci_mode=False)
np.set_printoptions(precision=2, suppress=True)

# MODEL_CKPT = '/home/deepvisionpoc/Desktop/Jeans/SOLIDER_exp/SOLIDER-PersonAttributeRecognition/exp_result/BagCountOxygenLabeller/swin_s.bc_labeller/img_model/from_stracht_mon_sk.pth'
MODEL_CKPT = '/home/deepvisionpoc/Desktop/Jeans/SOLIDER_exp/SOLIDER-PersonAttributeRecognition/exp_result/NoBP/swin_sNoBP/img_model/ckpt_max_2024-07-25_14:33:41lastEp.pth' 
FIG_PATH = '/home/deepvisionpoc/Desktop/Jeans/SOLIDER_exp/SOLIDER-PersonAttributeRecognition/fig'
model = BagPredictor(model_ckpt=MODEL_CKPT)

# Path to the root directory
root_dir = '/home/deepvisionpoc/Desktop/Jeans/resources/bag_by_store'

def process_folder(folder_path,save_path):
    image_paths = []
    pred_probs_list = []
    
    # Traverse the folder and get all image paths
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, file)
                image_paths.append(image_path)

    # Create subplots
    num_images = len(image_paths)
    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
    
    if num_images == 1:
        axes = [axes]

    for idx, image_path in enumerate(image_paths):
        # Run inference
        pred_class, pred_probs = model.run_inference(image_path)
        pred_probs_list.append(pred_probs)

        # Read and plot the image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        axes[idx].imshow(image)
        axes[idx].axis('off')
        axes[idx].set_title(f'Prob: {pred_probs}')
    
    # Save the plot
    # print(save_path)
    output_path = os.path.join(save_path, f'{"_".join(folder_path.split("/")[-2:])}.png')
    print(output_path)
    plt.savefig(output_path)
    plt.close()

# Process each subfolder in the root directory
for store_folder in os.listdir(root_dir):
    store_folder_path = os.path.join(root_dir, store_folder)
    if os.path.isdir(store_folder_path):
        for sub_folder in os.listdir(store_folder_path):
            sub_folder_path = os.path.join(store_folder_path, sub_folder)
            if os.path.isdir(sub_folder_path):
                process_folder(sub_folder_path,FIG_PATH)

print("Inference and plotting completed.")
