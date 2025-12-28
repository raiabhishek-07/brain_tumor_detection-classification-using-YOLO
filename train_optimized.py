from ultralytics import YOLO
import os

def train_optimized():
    # Load the model (nano version for speed, but pretrained)
    model = YOLO('yolov8n-cls.pt')

    # Define path to the new dataset (archive folder)
    # Ensure this path is absolute or relative from where script runs
    data_dir = os.path.join(os.getcwd(), 'archive')

    print(f"Starting Optimized Training on dataset at: {data_dir}")
    print("Optimization Strategy: AdamW + Cosine Annealing + Advanced Augmentation")

    # Train model with "Meta-Heuristic" style hyperparameters
    results = model.train(
        data=data_dir,
        epochs=15,             # Increased from 3 to 15 for better convergence
        imgsz=256,             # Standard MRI size
        batch=16,              # Standard batch size
        
        # Optimization
        optimizer='AdamW',     # Best generalization
        cos_lr=True,           # Cosine Annealing Scheduler (Research contribution)
        lr0=0.001,             # Initial LR
        lrf=0.01,              # Final LR factor
        
        # Regularization & Augmentation (Class-Aware equivalents)
        dropout=0.3,           # Reduce overfitting
        degrees=10.0,          # Rotation
        translate=0.1,         # Translation
        scale=0.5,             # Scaling
        fliplr=0.5,            # Left-Right Flip
        mosaic=1.0,            # Mosaic Augmentation (Strong feature learning)
        mixup=0.1,             # Mixup (Regularizer)
        
        # naming
        name='brain_tumor_optimized',
        project='runs/classify',
        exist_ok=True
    )
    
    # Validate
    print("Training Complete. Validating...")
    metrics = model.val()
    print(f"Top-1 Accuracy: {metrics.top1}")
    print(f"Top-5 Accuracy: {metrics.top5}")

if __name__ == '__main__':
    train_optimized()
