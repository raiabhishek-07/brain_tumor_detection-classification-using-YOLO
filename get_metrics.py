from ultralytics import YOLO
from sklearn.metrics import classification_report
import json
from pathlib import Path

# Load the OPTIMIZED model
model = YOLO('runs/classify/brain_tumor_optimized/weights/best.pt')

# Validation set path
data_dir = Path('archive/val')
class_names = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])

true_labels = []
pred_labels = []

print("Running inference on validation set...")
# Collect predictions
for i, class_name in enumerate(class_names):
    class_dir = data_dir / class_name
    for img_path in class_dir.glob('*.jpg'):
        # Predict
        results = model(str(img_path), verbose=False)
        top1 = results[0].probs.top1
        
        true_labels.append(i)
        pred_labels.append(top1)

# Generate Report
report = classification_report(true_labels, pred_labels, target_names=class_names, output_dict=True)

# Advanced Metrics Calculation
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score
import time
import os
import numpy as np

# 1. Computational Metrics
param_size = 0
for param in model.model.parameters():
    param_size += param.nelement() * param.element_size()
model_size_mb = param_size / 1024 / 1024

# 2. Inference Loop for Advanced Metrics
probs_list = []
true_list = []
inference_times = []

# Map classes to indices
class_map = model.names
reverse_map = {v: k for k, v in class_map.items()}

print("Calculating Advanced Metrics (ROC, LogLoss)...")
for img_file in data_dir.rglob('*.jpg'): # Adjust extension if needed
    if "checkpoint" in str(img_file): continue
    
    # Get True Label
    label_name = img_file.parent.name
    if label_name not in reverse_map: continue
    true_idx = reverse_map[label_name]
    true_list.append(true_idx)
    
    # Inference
    start_t = time.time()
    res = model.predict(img_file, verbose=False)[0]
    end_t = time.time()
    inference_times.append((end_t - start_t) * 1000) # ms
    
    # Probabilities
    probs = res.probs.data.cpu().numpy() # Softmax probs
    probs_list.append(probs)

# Convert to numpy
probs_array = np.array(probs_list)
true_array = np.array(true_list)

# 3. Calculate Scikit-Learn Metrics
try:
    log_loss_val = log_loss(true_array, probs_array)
    roc_auc_val = roc_auc_score(true_array, probs_array, multi_class='ovr')
except Exception as e:
    print(f"Error calc complex metrics: {e}")
    log_loss_val = 0.0
    roc_auc_val = 0.0

avg_inf_time = np.mean(inference_times)

# Update Report Dictionary
report['advanced'] = {
    'log_loss': log_loss_val,
    'roc_auc': roc_auc_val,
    'inference_time_ms': avg_inf_time,
    'model_size_mb': model_size_mb,
    'parameters': param_size
}

# Generate Confusion Matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

pred_array = np.argmax(probs_array, axis=1)
cm = confusion_matrix(true_array, pred_array)
class_names = [class_map[i] for i in range(len(class_map))]

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix (Optimized Model)')
plt.savefig('confusion_matrix_optimized.png')
print("Confusion Matrix saved as confusion_matrix_optimized.png")

# Print as JSON
print("\nJSON_REPORT_START")
print(json.dumps(report, indent=4))
print("JSON_REPORT_END")
