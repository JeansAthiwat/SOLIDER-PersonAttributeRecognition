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
from Omodel import swin_small_patch4_window7_224, QuantityClassifier, QuantityClassifierV2, QuantityClassifierV3, BaggageClassifier
from Odataset import create_GPT_train_test_loader, PersonWithBaggageDataset

from kornia.losses import binary_focal_loss_with_logits, focal_loss
# Function to save checkpoint

# Helper function to convert labels to one-hot encoding
def to_one_hot(labels, num_classes):
    return torch.eye(num_classes)[labels].to(labels.device)

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(run_name, filename))
    if is_best:
        torch.save(state, os.path.join(run_name, 'model_best.pth.tar'))

# Function to compute per-class accuracy
def compute_per_class_accuracy(y_true, y_pred, num_classes=4):
    class_correct = [0] * num_classes
    class_total = [0] * num_classes

    for i in range(len(y_true)):
        label = y_true[i]
        pred = y_pred[i]
        if label == pred:
            class_correct[label] += 1
        class_total[label] += 1

    out = ""
    for i in range(num_classes):
        if class_total[i] == 0:
            out += f'{i}: NO_SAMPLE |'
        else:
            out += f'| {i}: ({100 * class_correct[i] / class_total[i]:.2f}% [{class_correct[i]}/{class_total[i]}])  |'
    return out



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classes_weight=torch.tensor([1.49175036, 0.42052578, 1.08703607, 31.50757576], requires_grad=False).to(device)
# Device configuration
NUM_EPOCHS = 32
FREEZE_BACKBONE_EPOCHS = 0  # Number of epochs to freeze the backbone
BEST_MODEL_PATH = 'Oruns/GPT/focal_loss_WITH_weight/startAt41acc0718-13:02_E64_[1e-05,1e-05-min1e-06]/checkpoint.pth.tar'
TRAIN_FROM_SCRATCH = False
BATCH_SIZE = 64 #128
loss = "focal_loss"

# Define separate learning rates for backbone and classifier
backbone_lr = 1e-5
classifier_lr = 1e-5
eta_min = 1e-8
T_0 = NUM_EPOCHS//4
T_mult = 1

# Define the run name
run_name = f"Oruns/GPT/{loss}_WITH_weight/startAt44acc{datetime.now().strftime('%m%d-%H:%M')}_E{NUM_EPOCHS}_[{backbone_lr},{classifier_lr}-min{eta_min}]"
os.makedirs(run_name, exist_ok=True)

# Prepare the dataset & model
train_loader, test_loader = create_GPT_train_test_loader(BATCH_SIZE=BATCH_SIZE)
backbone = swin_small_patch4_window7_224()
classifier = QuantityClassifierV3()
model = BaggageClassifier(backbone, classifier).to(device)

if TRAIN_FROM_SCRATCH:
    model.backbone.load_state_dict(torch.load('results/pa100k/aqui_esta_par.pth'))
    print(f"Starting from Scratch")
else:
    ckpt = torch.load(BEST_MODEL_PATH)
    model.load_state_dict(ckpt['state_dict'])
    print(f"Loaded Model Details:\nEpoch: {ckpt['epoch']} Acc: {ckpt['best_acc']}")


# Freeze the backbone parameters
for param in model.backbone.parameters():
    param.requires_grad = False

# optimizer = SGD(model.parameters(), lr=lr, momentum=0.8)
optimizer = SGD([
    {'params': model.backbone.parameters(), 'lr': backbone_lr},
    {'params': model.classifier.parameters(), 'lr': classifier_lr}
], momentum=0.9)

scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min)
if loss.lower() == "focal_loss":
    criterion = lambda pred, target: focal_loss(pred, target, alpha=0.25, gamma=2.0, reduction='mean',weight=classes_weight)
    # criterion = lambda pred, target: focal_loss(pred, target, alpha=0.25, gamma=2.0, reduction='mean')
    print("Loss: FOCAL")
else:
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([ 1.30681492,  0.6247725 ,  1.09801359 ,6.00333333],requires_grad=False), label_smoothing=0.2).to(device)
    print("Loss: BalanceCE")
# criterion = torch.nn.
# from sklearn 
# Initialize TensorBoard writer
writer = SummaryWriter(log_dir=run_name)

# Training loop
best_acc = 0.0
print("train len : ", len(train_loader))
print("test len : ", len(test_loader))

for epoch in range(NUM_EPOCHS):
    if epoch == FREEZE_BACKBONE_EPOCHS:
        print("BACKBONE UNFROZEN")
        for param in model.backbone.parameters():
            param.requires_grad = True

    # Training phase
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    train_labels = []
    train_preds = []

    for images, targetTop1s, logProbTop1s, _, _, img_path in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{NUM_EPOCHS}"):
        images = images.to(device)
        targetTop1s = targetTop1s.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targetTop1s)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += targetTop1s.size(0)
        correct += predicted.eq(targetTop1s).sum().item()

        train_labels.extend(targetTop1s.cpu().numpy())
        train_preds.extend(predicted.cpu().numpy())

    train_loss = running_loss / len(train_loader.dataset)
    train_acc = correct / total

    # Compute per-class accuracy for training
    train_out = compute_per_class_accuracy(train_labels, train_preds)

    # Validation phase
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    val_labels = []
    val_preds = []

    with torch.no_grad():
        for images, targetTop1s, logProbTop1s, _, _, img_path in test_loader:
            images = images.to(device)
            # one_hot = to_one_hot(targetTop1s, 4).to(device)  # One-hot encode targets
            targetTop1s = targetTop1s.to(device)

            outputs = model(images)
            loss = criterion(outputs, targetTop1s)

            val_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += targetTop1s.size(0)
            correct += predicted.eq(targetTop1s).sum().item()

            val_labels.extend(targetTop1s.cpu().numpy())
            val_preds.extend(predicted.cpu().numpy())

    val_loss /= len(test_loader.dataset)
    val_acc = correct / total

    # Compute per-class accuracy for validation
    test_out = compute_per_class_accuracy(val_labels, val_preds)
    
    print("Training Per-Class Accuracy   :", train_out)
    print("Validation Per-Class Accuracy :", test_out)


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
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] lr:[{optimizer.param_groups[0]['lr']:.1e},{optimizer.param_groups[1]['lr']:.1e}] Train Loss: {train_loss:.4f} Train Acc: {train_acc:.4f} Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f}")

    # Update the learning rate
    scheduler.step()

writer.close()
