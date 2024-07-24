import os
import pickle
import numpy as np
from easydict import EasyDict as edict
from tools.function import get_pkl_rootpath
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance

def get_wrong_pred(gt_label, preds_probs, path_list):
    
    gt_class = np.argmax(gt_label,axis=1)
    pred_class = np.argmax(preds_probs,axis=1)
    # print(gt_class)
    # print(pred_class)
    # for i in range(gt_label.shape[0]):
    #     print("--------------------------")
    #     print("GT   : " ,gt_label[i])
    #     print("Pred : " ,preds_probs[i])
    #     print("--------------------------")
    incorrect_classifications = (gt_class != pred_class)
    # print(incorrect_classifications)
    
    wrong_pred_path_list = [path_list[i] for i, incorrect in enumerate(incorrect_classifications) if incorrect]
    
    print(f"Total Incorrect : {len(wrong_pred_path_list)}/{len(path_list)} Images")
    wrong_pred_class = pred_class[incorrect_classifications]    
    wrong_preds_probs = preds_probs[incorrect_classifications]
    wrong_preds_gt_label = gt_label[incorrect_classifications]
    
    return wrong_pred_path_list, wrong_pred_class, wrong_preds_probs, wrong_preds_gt_label

def save_wrong_pred_figure(cfg, gt_label, wrong_pred_path_list, wrong_pred_class, wrong_preds_probs, fig_dir='Incorrect_fig_output'):
    # Ensure the figure directory exists
    np.set_printoptions(precision=2, suppress=True)
    fold_name = cfg.DATASET.TEST_SPLIT
    dataset_name = cfg.DATASET.NAME
    fig_dir = os.path.join(fig_dir,dataset_name,fold_name)
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    
    pkl_path = get_pkl_rootpath(cfg.DATASET.NAME, cfg.DATASET.ZERO_SHOT)
    print("which pickle", pkl_path)    

    dataset_info = pickle.load(open(pkl_path, 'rb+'))
    root_path = f"./data/{dataset_info.root[38:]}"  # Adjust the root path

    gt_class = np.argmax(gt_label, axis=1)
    pred_class = np.argmax(wrong_preds_probs,axis=1)
    num_images = len(wrong_pred_path_list)
    num_images_per_row = 4
    num_images_per_figure = num_images_per_row * 4  # 4 rows per figure
    

    for start_idx in range(0, num_images, num_images_per_figure):
        fig, axes = plt.subplots(4, num_images_per_row, figsize=(12, 20))
        # fig.subplots_adjust(hspace=0.3, wspace=0.3)  # Add horizontal space between plots
        
        for i in range(num_images_per_figure):
            idx = start_idx + i
            if idx >= num_images:
                break
            
            img_path = os.path.join(root_path, wrong_pred_path_list[idx])
            image = Image.open(img_path)
            
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(0.7)
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(1.3)

            
            ax = axes[i // num_images_per_row, i % num_images_per_row]
            ax.imshow(image)
            ax.axis('off')
            gt = gt_label[idx]
            pred = wrong_pred_class[idx]
            prob = wrong_preds_probs[idx][pred]
            
            # Truncate the file path if it's too long
            truncated_path = wrong_pred_path_list[idx] if len(wrong_pred_path_list[idx]) <= 20 else f'...{wrong_pred_path_list[idx][-17:]}'
            
            # ax.set_title(f'{truncated_path}\nGT: {gt}\nPred: {pred} Prob: {prob:.2f}', fontsize=8)  # Reduced font size
            ax.set_title(f'{idx} {truncated_path}\nGT  : {gt_class[idx]} {gt}\nProb: {pred_class[idx]} {wrong_preds_probs[idx]}', fontsize=8)  # Reduced font size
       
        
        fig_file_name = f'wrong_pred_{fold_name}_{start_idx // num_images_per_figure}.png' if fold_name else f'wrong_pred_{start_idx // num_images_per_figure}.png'
        fig_path = os.path.join(fig_dir, fig_file_name)
        plt.savefig(fig_path)
        plt.close(fig)
