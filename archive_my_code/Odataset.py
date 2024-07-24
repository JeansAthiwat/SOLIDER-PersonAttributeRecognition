import torch
import torch.nn as nn
import timm
from torchvision.transforms import v2 as T
import os
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.model_selection import train_test_split
from PIL import Image
import csv
import cv2
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import pickle

INPUT_IMAGES_SIZE = (224, 224)
TRAIN_CSV_FILE = "/home/deepvisionpoc/Desktop/Jeans/SOLIDER_exp/SOLIDER-PersonAttributeRecognition/Omanifest/train_manifest.csv"
TEST_CSV_FILE = "/home/deepvisionpoc/Desktop/Jeans/SOLIDER_exp/SOLIDER-PersonAttributeRecognition/Omanifest/test_manifest.csv"
ROOT_DIR = "/home/deepvisionpoc/Desktop/Jeans/resources/mon"


def parse_pickle(pickle_path):
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    return data

def filter_token(all_prediction, top_k=3):
    filtered_list = []
    for idx , pred in enumerate(all_prediction):
        if pred['token'].isdigit() and 0 <= int(pred['token']) <= 10:
            filtered_list.append(pred)
    return filtered_list[:top_k]

def get_images_top_k_pred(image_key,data, top_k = 3):
    try:
        all_prediction = data[image_key]['choices'][0]['logprobs']['content'][0]['top_logprobs'] # get top 5 log prob
        top_k_predict = filter_token(all_prediction,top_k)
    except Exception as e:
        raise e
    return top_k_predict
def normalize(probs):
    return probs / np.sum(probs)
# Define the transformations for the training set
TRAIN_TRANSFORM = T.Compose(
    [
        T.Resize(INPUT_IMAGES_SIZE),
        T.RandomApply(
            [
                T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
                T.RandomPerspective(distortion_scale=0.20, p=0.8),
                # T.RandomAdjustSharpness(sharpness_factor=1.5, p=0.5),
                T.RandomHorizontalFlip(),
            ],
            p=0.99,
        ),  # Apply these transformations with a probability of 0.9
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Define the transformations for the validation set
VAL_TRANSFORM = T.Compose(
    [
        T.Resize(INPUT_IMAGES_SIZE),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

GPT_PKL = 'data/BagCountGPT/all_responses_new.pkl'
GPT_ROOT = 'data/BagCountGPT/image-samples-by-class-flattened'

class GPTDataset(Dataset):
    def __init__(self, pickle_file=GPT_PKL, root_dir=GPT_ROOT, transform=None, use_expected_value=True, max_bag = 3):
        self.data = parse_pickle(pickle_file)
        self.root_dir = root_dir
        self.transform = transform
        self.use_expected_value = use_expected_value
        
        self.images_name = []
        self.images_tokens = []
        self.images_tokens_top1 = []
        self.images_logprobs = []
        self.images_expected_value = []
        for image_name in list(self.data.keys()):
            try:
                top_k_predict = get_images_top_k_pred(image_name, self.data, top_k=3)

                # tmp_token = [min(3, int(pred['token'])) for pred in top_k_predict]
                tmp_logprob = [float(pred['logprob']) for pred in top_k_predict]
                tmp_token = [min(3, int(token)) for pred in top_k_predict for token in pred['token']]
                
                prob = np.exp(tmp_logprob)
                prob_norm = normalize(prob)
                expected_value = np.sum([min(3, round(t * p)) for t, p in zip(tmp_token, prob_norm)])
                
                self.images_tokens.append(tmp_token)
                self.images_tokens_top1.append(tmp_token[0])
                self.images_logprobs.append(tmp_logprob)
                self.images_name.append(image_name)   
                self.images_expected_value.append(expected_value)
            except Exception as e:
                # print(f"Error processing {image_name}: {e}")
                continue
            
    def __len__(self):
        return len(self.images_name)
    
    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image_name = self.images_name[idx]
        img_path = os.path.join(self.root_dir, image_name)
        img = Image.open(img_path).convert("RGB")
        
        if self.transform:
            img = self.transform(img)
        
        targetTop1, logProbTop1 = self.images_tokens_top1[idx], self.images_logprobs[idx][0]
        targetEV = self.images_expected_value[idx]

        return img, targetTop1, logProbTop1, targetEV, [self.images_tokens[idx], self.images_logprobs[idx]] , img_path
            
def create_GPT_train_test_loader(TEST_SIZE=0.15, BATCH_SIZE=64, SEED=42):
    # Create the GPTDataset with training transformations
    gpt_train_dataset = GPTDataset(transform=TRAIN_TRANSFORM)
    # Create the GPTDataset with validation transformations
    gpt_val_dataset = GPTDataset(transform=VAL_TRANSFORM)
    
    # Split the dataset indices for training and testing
    train_indices, test_indices, _, _ = train_test_split(
        range(len(gpt_train_dataset)),
        gpt_train_dataset.images_tokens_top1,
        stratify=gpt_train_dataset.images_tokens_top1,    
        test_size=TEST_SIZE,
        random_state=SEED
    )
    
    # Create subsets using the train and test indices
    train_split = Subset(gpt_train_dataset, train_indices)
    test_split = Subset(gpt_val_dataset, test_indices)
    
    # Create DataLoaders for the train and test splits
    train_loader = DataLoader(train_split, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_split, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, test_loader

        
class PersonWithBaggageDataset(Dataset):
    def __init__(self, csv_file, root_dir=ROOT_DIR, transform=None):
        self.transform = transform
        self.images = []
        self.labels = []

        with open(csv_file, "r") as csvfile:
            csvreader = csv.reader(csvfile)
            # next(csvreader)  # Skip header
            for row in csvreader:
                img_path, label = row
                img_path = os.path.join(root_dir, img_path)
                self.images.append(img_path)
                self.labels.append(int(label))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_path ,label = self.images[idx], self.labels[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        return img, label

        
# class CentralSamePersonPair(Dataset):
#     def __init__(self, csv_pair_file, root_dir=ROOT_DIR, transform=None):
#         self.transform = transform
#         self.image_pairs = []
#         self.label_pairs = []

#         # Read CSV file and populate images and labels
#         with open(csv_pair_file, "r") as csvfile:
#             csvreader = csv.reader(csvfile)
#             next(csvreader)  # Skip header?
#             for row in csvreader:
#                 img1_path, img2_path, label1, label2 = row
#                 img1_path = os.path.join(root_dir, img1_path)
#                 img2_path = os.path.join(root_dir, img2_path)
#                 self.image_pairs.append([img1_path, img2_path])
#                 self.label_pairs.append([int(label1), int(label2)])

#     def __len__(self):
#         return len(self.image_pairs)

#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()
            
#         img1_path, img2_path = self.image_pairs[idx]
#         label1, label2 = self.label_pairs[idx]

#         # Load images
#         img1 = Image.open(img1_path).convert("RGB")
#         img2 = Image.open(img2_path).convert("RGB")
        
#         if self.transform:
#             img1 = self.transform(img1)
#             img2 = self.transform(img2)

#         return img1, img2, label1, label2

# Example usage:
if __name__ == "__main__":
    csv_file = TRAIN_CSV_FILE
    dataset = PersonWithBaggageDataset(csv_file, ROOT_DIR,TRAIN_TRANSFORM)

    # Extract all labels
    all_labels = [dataset[i][1] for i in range(len(dataset))]

    # Compute class weights
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(all_labels), y=all_labels)
    class_weights_dict = {i: class_weights[i] for i in np.unique(all_labels)}

    print("Class Weights:", class_weights_dict)

    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    for batch_index, (imgs, labels) in enumerate(dataloader):
        print(batch_index, imgs.shape, labels.shape)
        break  # Display only the first batch for brevity

