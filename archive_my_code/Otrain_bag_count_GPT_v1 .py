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
from Omodel import swin_small_patch4_window7_224, QuantityClassifier,QuantityClassifierV2, BaggageClassifier
from Odataset import PersonWithBaggageDataset,GPTDataset ,TRAIN_CSV_FILE, TEST_CSV_FILE, ROOT_DIR, TRAIN_TRANSFORM, VAL_TRANSFORM
from sklearn.utils.class_weight import compute_class_weight

# Function to save checkpoint
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(run_name, filename))
    if is_best:
        torch.save(state, os.path.join(run_name, 'model_best.pth.tar'))

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_EPOCHS = 86
FREEZE_BACKBONE_EPOCHS = 6  # Number of epochs to freeze the backbone
BEST_MODEL_PATH = '/home/deepvisionpoc/Desktop/Jeans/SOLIDER_exp/SOLIDER-PersonAttributeRecognition/Oruns/GPT/balanceCE20240716_162445_56epochs_[backbone_lr1e-05,classifier_lr0.001, eta_min1e-06]/model_best.pth.tar'
TRAIN_FROM_SCRATCH = True

# Define separate learning rates for backbone and classifier
backbone_lr = 5e-5
classifier_lr = 1e-5
eta_min = 1e-6
T_0 = NUM_EPOCHS
T_mult = 2
# Define the run name
run_name = f"Oruns/GPT/CE_sameLR{datetime.now().strftime('%Y%m%d_%H%M%S')}_{NUM_EPOCHS}epochs_[backbone_lr{backbone_lr},classifier_lr{classifier_lr}, eta_min{eta_min}]"
os.makedirs(run_name, exist_ok=True)

# Prepare the dataset
train_ds = GPTDataset(transform=TRAIN_TRANSFORM,use_expected_value=True)
# Extract all labels
# all_labels = [train_ds[i][1] for i in range(len(train_ds))]
# class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(all_labels), y=all_labels)
# class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)
# print("class_weights_tensor:", class_weights_tensor)

test_ds = PersonWithBaggageDataset(TEST_CSV_FILE, ROOT_DIR, transform=VAL_TRANSFORM)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

# Prepare the model
backbone = swin_small_patch4_window7_224()
classifier = QuantityClassifierV2()
model = BaggageClassifier(backbone, classifier).to(device)
if TRAIN_FROM_SCRATCH:
    model.backbone.load_state_dict(torch.load('results/pa100k/aqui_esta_par.pth'))
else:
    ckpt = torch.load(BEST_MODEL_PATH)
    model.load_state_dict(ckpt['state_dict'])
    print(f"Loaded Model Details:\nEpoch: {ckpt['epoch']} Acc: {ckpt['best_acc']}")
    
# Freeze the backbone parameters
for param in model.backbone.parameters():
    param.requires_grad = False


# optimizer = SGD([
#     {'params': model.backbone.parameters(), 'lr': backbone_lr},
#     {'params': model.classifier.parameters(), 'lr': classifier_lr}
# ], momentum=0.8)
optimizer = SGD(model.parameters(),lr=backbone_lr, momentum=0.8)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min)
criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.4).to(device)


# Initialize TensorBoard writer
writer = SummaryWriter(log_dir=run_name)

# Training loop
best_acc = 0.0
print("train len", len(train_loader))
for epoch in range(NUM_EPOCHS):
    # Unfreeze the backbone after FREEZE_BACKBONE_EPOCHS
    if epoch == FREEZE_BACKBONE_EPOCHS:
        print("BACKBONE UNFROZEN")
        for param in model.backbone.parameters():
            param.requires_grad = True
            
    # Training phase
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # for images, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{NUM_EPOCHS}"):
    for images, labels , _ , _ , _ in tqdm(train_loader,desc=f"Training: "):
        images = images.to(device)
        labels = labels.to(device)
        # print("EV", labels)

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
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] lr:[{optimizer.param_groups[0]['lr']:.1e}] Train Loss: {train_loss:.4f} Train Acc: {train_acc:.4f} Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f}")

    # Update the learning rate
    scheduler.step()

writer.close()
