"""
Inference script for Brain Tumor Detection using trained YOLOv8 Classification model
"""
from ultralytics import YOLO
import random
from pathlib import Path
import os
import shutil

# Configuration
# Path to the best trained model (will be created after training)
model_path = Path('runs/classify/brain_tumor_yolo_fast/weights/best.pt')
data_dir = Path('data/val') # Using validation set which we renamed from 'Testing'

def get_random_image():
    # Provide backward compatibility if folders haven't been renamed yet
    search_dir = data_dir if data_dir.exists() else Path('data/Testing')
    
    all_images = list(search_dir.glob('*/*.jpg'))
    if not all_images:
        print(f"No images found in {search_dir}")
        exit(1)
    choice = random.choice(all_images)
    return choice

def predict_image(image_path):
    print(f"\nSelected Image: {image_path.name}")
    print(f"True Class: {image_path.parent.name}")
    
    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        print("Please wait for training to complete.")
        return

    # Load model
    print("Loading YOLOv8 model...")
    model = YOLO(model_path)
    
    # Predict
    print("-" * 30)
    # predict() returns a list of Results objects
    results = model.predict(image_path)
    
    # Process results
    result = results[0]
    probs = result.probs  # Probs object
    
    # Get top 1 prediction
    top1_index = probs.top1
    top1_conf = probs.top1conf.item()
    
    # In YOLO classification, names map is in result.names
    pred_class_name = result.names[top1_index]
    
    print(f"PREDICTION: {pred_class_name}")
    print(f"Confidence: {top1_conf * 100:.2f}%")
    print("-" * 30)
    
    if pred_class_name == image_path.parent.name:
        print("✅ Correct Prediction")
    else:
        print("❌ Incorrect Prediction")

if __name__ == "__main__":
    try:
        img_path = get_random_image()
        predict_image(img_path)
    except Exception as e:
        print(f"Error: {e}")
