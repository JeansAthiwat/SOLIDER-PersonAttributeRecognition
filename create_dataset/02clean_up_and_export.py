import os
import shutil
import cv2
import csv

DATA_DATE = '2024-04-12'
CHUNK_NUM = 2
LABELED_ROOT = f'/home/deepvisionpoc/Desktop/Jeans/resources/bag_count/labeled/{DATA_DATE}_chunk_{CHUNK_NUM}'
# DEST_ROOT = f'/home/deepvisionpoc/Desktop/Jeans/resources/bag_count/cleaned/{DATA_DATE}_chunk_{CHUNK_NUM}'
DEST_ROOT = f'/home/deepvisionpoc/Desktop/Jeans/resources/bag_count/cleaned/MallBagOnly'

os.makedirs(DEST_ROOT,exist_ok=True)
LABELED_CSV = os.path.join(DEST_ROOT, 'label_progress.csv')

# Load labeled images from CSV
counts = [0,0,0,0]

labeled_imgs = dict()
if os.path.exists(LABELED_CSV):
    with open(LABELED_CSV, 'r') as csv_f:
        reader = csv.reader(csv_f)
        for row in reader:
            if row:  # check if row is not empty
                img_name, current_status = row
                print(current_status)
                assert not img_name == "", "Bro its empty row"
                assert not img_name in labeled_imgs.keys(), f"fuck me its : {img_name}"
                labeled_imgs[img_name] = current_status
                if current_status.isdigit():
                    counts[int(current_status)] += 1
print(counts)
breakpoint()
# Process images
rem_count =0 
dict_keys = labeled_imgs.keys()
for root, dirs, files in os.walk(LABELED_ROOT):
    for file in files:

        if not file.endswith(('.jpg', 'png')):
            continue
        
        img_path = os.path.join(root, file)
        image_name = os.path.basename(img_path)

        if (image_name in dict_keys) and (not ((labeled_imgs[image_name] == "delete") or (labeled_imgs[image_name] == "rare" ))):
            label = int(img_path.split('/')[-2])
            shutil.copy(img_path, os.path.join(DEST_ROOT,str(label)) )

        else:
            # os.remove(img_path)
            rem_count += 1
            # print("removed :", img_path)
print(rem_count)


