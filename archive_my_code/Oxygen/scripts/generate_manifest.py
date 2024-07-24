import os
import csv

# Define the path to the main directory
main_dir = '/home/deepvisionpoc/Desktop/Jeans/resources/mon'

# Define the chunk ranges for training and testing
train_chunks = ['2024-06-22_chunk_{}'.format(i) for i in range(0, 7)]
test_chunks = ['2024-06-22_chunk_{}'.format(i) for i in range(7, 9)]

# Define the label mapping
label_mapping = {
    '0': 0,
    '1': 1,
    '2': 2,
    '3': 3,
    '4': 3,
    '5': 3
}

def create_manifest(manifest_file, chunks):
    with open(manifest_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        
        for chunk in chunks:
            chunk_dir = os.path.join(main_dir, chunk)
            
            for label in label_mapping.keys():
                label_dir = os.path.join(chunk_dir, label)
                
                if os.path.exists(label_dir):
                    for img_file in os.listdir(label_dir):
                        if img_file.endswith('.jpg'):
                            relative_path = os.path.join(chunk, label, img_file)
                            writer.writerow([relative_path, label_mapping[label]])

# Create training manifest file
create_manifest('train_manifest.csv', train_chunks)

# Create testing manifest file
create_manifest('test_manifest.csv', test_chunks)
