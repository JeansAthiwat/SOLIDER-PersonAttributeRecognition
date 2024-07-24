import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from Omodel import swin_small_patch4_window7_224, QuantityClassifier, BaggageClassifier
from Odataset import PersonWithBaggageDataset,CentralSamePersonPair, TRAIN_CSV_FILE, TEST_CSV_FILE, ROOT_DIR, TRAIN_TRANSFORM, VAL_TRANSFORM

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_EPOCHS = 100
BEST_MODEL_PATH = '/home/deepvisionpoc/Desktop/Jeans/SOLIDER_exp/SOLIDER-PersonAttributeRecognition/Oxygen_runs/run_20240712_152253/model_best.pth.tar'  # Update this path to your best model file

# Prepare the dataset
train_ds = CentralSamePersonPair(TRAIN_CSV_FILE, ROOT_DIR, TRAIN_TRANSFORM)
test_ds = CentralSamePersonPair(TEST_CSV_FILE, ROOT_DIR, VAL_TRANSFORM)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

# Prepare the model
backbone = swin_small_patch4_window7_224()
classifier = QuantityClassifier()
model = BaggageClassifier(backbone, classifier).to(device)
model.load_state_dict(torch.load(BEST_MODEL_PATH)['state_dict'])  # Load the best model
# Define separate learning rates for backbone and classifier
backbone_lr = 1e-5
classifier_lr = 1e-5

# Prepare optimizer with separate learning rates
optimizer = SGD([
    {'params': model.backbone.parameters(), 'lr': backbone_lr},
    {'params': model.classifier.parameters(), 'lr': classifier_lr}
], momentum=0.8)

# Prepare scheduler
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=NUM_EPOCHS // 5, T_mult=1, eta_min=1e-6)

# Prepare criterion
criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.05).to(device)

# Define the run name
run_name = f"Oxygen_runs/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(run_name, exist_ok=True)

# Initialize TensorBoard writer
writer = SummaryWriter(log_dir=run_name)

# Function to save checkpoint
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(run_name, filename))
    if is_best:
        torch.save(state, os.path.join(run_name, 'model_best.pth.tar'))

# Training loop
best_acc = 0.0

for epoch in range(NUM_EPOCHS):
    # Training phase
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{NUM_EPOCHS}"):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_loss = running_loss / len(train_loader.dataset)
    train_acc = correct / total

    # Validation phase
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    val_loss /= len(test_loader.dataset)
    val_acc = correct / total

    # Save the best model checkpoint
    is_best = val_acc > best_acc
    best_acc = max(val_acc, best_acc)
    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'best_acc': best_acc,
        'optimizer': optimizer.state_dict(),
    }, is_best)

    # Log to TensorBoard
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Loss/val', val_loss, epoch)
    writer.add_scalar('Accuracy/train', train_acc, epoch)
    writer.add_scalar('Accuracy/val', val_acc, epoch)

    # Print progress
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] lr:[{optimizer.param_groups[0]['lr']:.3e},{optimizer.param_groups[1]['lr']:.3e}] Train Loss: {train_loss:.4f} Train Acc: {train_acc:.4f} Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f}")

    # Print learning rates
    # for param_group in optimizer.param_groups:
    #     print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Learning Rate: {param_group['lr']}")

    # Update the learning rate
    scheduler.step()

writer.close()
