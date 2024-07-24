import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report
from Omodel import swin_small_patch4_window7_224, QuantityClassifier, BaggageClassifier
from Odataset import PersonWithBaggageDataset, TEST_CSV_FILE, ROOT_DIR, VAL_TRANSFORM

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path to the best model checkpoint
BEST_MODEL_PATH = 'Oruns/new_ds/1mlp_BalanceCE_SixthRun_07-17_13-59_e60_[1e-05,0.003]_lrmin6e-06/checkpoint.pth.tar'

# Load the test dataset
test_ds = PersonWithBaggageDataset(TEST_CSV_FILE, ROOT_DIR, VAL_TRANSFORM)
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=8, pin_memory=True)

# Load the model
backbone = swin_small_patch4_window7_224()
classifier = QuantityClassifier()
model = BaggageClassifier(backbone, classifier).to(device)

# Load the best model checkpoint
ckpt = torch.load(BEST_MODEL_PATH)
model.load_state_dict(ckpt['state_dict'])
model.eval()

# Function to compute per-class accuracy
def compute_per_class_accuracy(y_true, y_pred, num_classes):
    class_correct = [0] * num_classes
    class_total = [0] * num_classes

    for i in range(len(y_true)):
        label = y_true[i]
        pred = y_pred[i]
        if label == pred:
            class_correct[label] += 1
        class_total[label] += 1

    for i in range(num_classes):
        if class_total[i] == 0:
            print(f'Class {i} has no samples in the test set.')
        else:
            print(f'Accuracy of class {i}: {100 * class_correct[i] / class_total[i]:.2f}% [{class_correct[i]} / {class_total[i]}]')

# Perform inference and compute per-class accuracy
all_labels = []
all_predictions = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = outputs.max(1)

        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

# Calculate and print the per-class accuracy
num_classes = len(np.unique(all_labels))
compute_per_class_accuracy(all_labels, all_predictions, num_classes)