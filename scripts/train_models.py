import os
import argparse
from pathlib import Path
from ultralytics import YOLO
import yaml
import shutil


def train_model(data_yaml, output_dir, model_type='yolov8n.pt', epochs=50, batch_size=16, imgsz=640, model_name='model'):
    """
    Train a YOLOv8 model
    
    Parameters:
    data_yaml (str): Path to data.yaml file
    output_dir (Path): Path to save the model weights
    model_type (str): Base model type
    epochs (int): Number of training epochs
    batch_size (int): Batch size
    imgsz (int): Image size
    model_name (str): Name for the saved model
    
    Returns:
    Path: Path to the best weights file
    """
    print(f"Starting training for {model_name} detection model")
    
    # Create model
    model = YOLO(model_type)
    
    # Set up training arguments
    args = {
        'data': data_yaml,
        'epochs': epochs,
        'patience': 10,
        'batch': batch_size,
        'imgsz': imgsz,
        'device': 'cpu',  # Use GPU if available
        'name': model_name,
        'project': str(output_dir),
        'exist_ok': True,
        'pretrained': True,
        'optimizer': 'Adam',
        'verbose': True,
        'seed': 42,
        'deterministic': True,
        'val': True,
        'save': True
    }
    
    # Train the model
    results = model.train(**args)
    
    # Get path to best weights
    best_weights = Path(output_dir) / model_name / 'weights' / 'best.pt'
    
    # Copy best weights to a more descriptive filename
    final_weights = Path(output_dir) / f"{model_name}_best.pt"
    shutil.copy(best_weights, final_weights)
    
    print(f"Training complete. Best weights saved to {final_weights}")
    
    return final_weights


def main():
    parser = argparse.ArgumentParser(description='Train person and PPE detection models')
    parser.add_argument('dataset_dir', type=str, help='Directory containing prepared datasets')
    parser.add_argument('output_dir', type=str, help='Directory to save model weights')
    parser.add_argument('--person_model', type=str, default='yolov8n.pt', help='Base model for person detection')
    parser.add_argument('--ppe_model', type=str, default='yolov8n.pt', help='Base model for PPE detection')
    parser.add_argument('--person_epochs', type=int, default=50, help='Number of epochs for person detection')
    parser.add_argument('--ppe_epochs', type=int, default=50, help='Number of epochs for PPE detection')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--person_only', action='store_true', help='Train only person detection model')
    parser.add_argument('--ppe_only', action='store_true', help='Train only PPE detection model')
    
    args = parser.parse_args()
    
    # Setup paths
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if person model should be trained
    if not args.ppe_only:
        person_data_yaml = dataset_dir / 'person_detection/data.yaml'
        if not person_data_yaml.exists():
            print(f"Error: Person detection data.yaml not found at {person_data_yaml}")
            return
        
        person_model_path = train_model(
            data_yaml=str(person_data_yaml),
            output_dir=output_dir,
            model_type=args.person_model,
            epochs=args.person_epochs,
            batch_size=args.batch_size,
            imgsz=args.imgsz,
            model_name='person_detection'
        )
        
        # Create a weights directory and copy the best weights
        weights_dir = output_dir / 'weights'
        os.makedirs(weights_dir, exist_ok=True)
        shutil.copy(person_model_path, weights_dir / 'person_detection.pt')
    
    # Check if PPE model should be trained
    if not args.person_only:
        ppe_data_yaml = dataset_dir / 'ppe_detection/data.yaml'
        if not ppe_data_yaml.exists():
            print(f"Error: PPE detection data.yaml not found at {ppe_data_yaml}")
            return
        
        ppe_model_path = train_model(
            data_yaml=str(ppe_data_yaml),
            output_dir=output_dir,
            model_type=args.ppe_model,
            epochs=args.ppe_epochs,
            batch_size=args.batch_size,
            imgsz=args.imgsz,
            model_name='ppe_detection'
        )
        
        # Copy the best weights to the weights directory
        weights_dir = output_dir / 'weights'
        os.makedirs(weights_dir, exist_ok=True)
        shutil.copy(ppe_model_path, weights_dir / 'ppe_detection.pt')
    
    print("Training complete! Model weights saved to the 'weights' directory.")


if __name__ == "__main__":
    main()