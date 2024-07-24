import os
import shutil
import cv2
import csv

DATA_DATE = '2024-04-12'
CHUNK_NUM = 0
LABELED_ROOT = f'/home/deepvisionpoc/Desktop/Jeans/resources/bag_count/labeled/{DATA_DATE}_chunk_{CHUNK_NUM}'
LABELED_CSV = os.path.join(LABELED_ROOT, 'label_progress.csv')
CURRENT_CLASS = '3'

# Load labeled images from CSV
labeled_imgs = set()
if os.path.exists(LABELED_CSV):
    with open(LABELED_CSV, 'r') as csv_f:
        reader = csv.reader(csv_f)
        for row in reader:
            if row:  # check if row is not empty
                img_name, current_status = row
                labeled_imgs.add(img_name)


def update_csv(image_name, status):
    labeled_imgs.add(image_name)
    with open(LABELED_CSV, 'a', newline='') as csv_f:  # 'a' mode for appending
        writer = csv.writer(csv_f)
        writer.writerow([image_name, status])

# Process images
for root, dirs, files in os.walk(os.path.join(LABELED_ROOT, CURRENT_CLASS)):
    for file in files:
        if not file.endswith(('.jpg', 'png')):
            continue
        
        img_path = os.path.join(root, file)
        image_name = os.path.basename(img_path)
        
        if image_name in labeled_imgs:
            continue
        
        # Open image using OpenCV
        image = cv2.imread(img_path)
        
        # Resize image to fill the top and bottom
        screen_height = 1080  # Assuming a common screen height if windo1w not visible
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
            update_csv(image_name, 'delete')
            print(f"MARKED AS TO DELETE: {image_name}")
        elif key == ord('k'):  # Keep
            update_csv(image_name, CURRENT_CLASS)
            print(f"OK: {image_name}")
        elif key == ord('0'):  # Move to class 0
            new_path = os.path.join(LABELED_ROOT, '0', image_name)
            if not os.path.exists(os.path.join(LABELED_ROOT, '0')):
                os.makedirs(os.path.join(LABELED_ROOT, '0'))
            shutil.move(img_path, new_path)
            update_csv(image_name, '0')
            print(f"MOVED TO CLASS 0: {image_name}")
        elif key == ord('1'):  # Move to class 1
            new_path = os.path.join(LABELED_ROOT, '1', image_name)
            if not os.path.exists(os.path.join(LABELED_ROOT, '1')):
                os.makedirs(os.path.join(LABELED_ROOT, '1'))
            shutil.move(img_path, new_path)
            update_csv(image_name, '1')
            print(f"MOVED TO CLASS 1: {image_name}")
        elif key == ord('2'):  # Move to class 2
            new_path = os.path.join(LABELED_ROOT, '2', image_name)
            if not os.path.exists(os.path.join(LABELED_ROOT, '2')):
                os.makedirs(os.path.join(LABELED_ROOT, '2'))
            shutil.move(img_path, new_path)
            update_csv(image_name, '2')
            print(f"MOVED TO CLASS 2: {image_name}")
        elif key == ord('3'):  # Move to class 3
            new_path = os.path.join(LABELED_ROOT, '3', image_name)
            if not os.path.exists(os.path.join(LABELED_ROOT, '3')):
                os.makedirs(os.path.join(LABELED_ROOT, '3'))
            shutil.move(img_path, new_path)
            update_csv(image_name, '3')
            print(f"MOVED TO CLASS 3: {image_name}")
        
        cv2.destroyAllWindows()
