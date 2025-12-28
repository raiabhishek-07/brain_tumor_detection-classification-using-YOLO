"""
Training script for Brain Tumor Detection using YOLOv8 Classification
"""
from ultralytics import YOLO
import os
import shutil
from pathlib import Path

print("=" * 60)
print("YOLOv8 BRAIN TUMOR CLASSIFICATION TRAINING")
print("=" * 60)

# ============================================================================
# 1. SETUP DATASET STRUCTURE
# ============================================================================
print("\n[1/3] Verifying Dataset Structure...")
data_path = Path('data')
training_path = data_path / 'Training'
testing_path = data_path / 'Testing'
train_path = data_path / 'train'
val_path = data_path / 'val'

# Rename 'Training' -> 'train' if needed
if training_path.exists() and not train_path.exists():
    print(f"Renaming '{training_path}' to '{train_path}' for YOLOv8 compatibility...")
    os.rename(training_path, train_path)
elif train_path.exists():
    print(f"'{train_path}' directory already exists. Good.")
else:
    print(f"ERROR: Could not find training data at '{training_path}' or '{train_path}'")
    exit(1)

# Rename 'Testing' -> 'val' if needed
if testing_path.exists() and not val_path.exists():
    print(f"Renaming '{testing_path}' to '{val_path}' for YOLOv8 compatibility...")
    os.rename(testing_path, val_path)
elif val_path.exists():
    print(f"'{val_path}' directory already exists. Good.")
else:
    print(f"Warning: Could not find testing data at '{testing_path}' or '{val_path}'")

print("Dataset structure is ready.")

# ============================================================================
# 2. INITIALIZE MODEL
# ============================================================================
print("\n[2/3] Initializing YOLOv8n-cls Model...")
# Load a pretrained YOLOv8n-cls model (nano version, pretrained on ImageNet)
model = YOLO('yolov8n-cls.pt')

print("Model initialized.")

# ============================================================================
# 3. TRAIN MODEL
# ============================================================================
print("\n[3/3] Starting Training...")
print("Training for 20 epochs. Logs will be saved to 'runs/classify/train'")

# Train the model
# data: path to dataset directory containing 'train' and 'val'
# epochs: number of training epochs
# imgsz: input image size (128x128 to match previous attempt)
results = model.train(
    data='data',
    epochs=3,
    imgsz=128,
    project='runs/classify',
    name='brain_tumor_yolo_fast'
)

print("\n" + "=" * 60)
print("TRAINING COMPLETE!")
print("=" * 60)
print(f"Results saved to: {results.save_dir}")
print("You can now run 'python test_yolo.py' (to be created) for inference.")
