#!/usr/bin/env python3
"""
YOLOv8 Training Script for Pool Balls Detection
===============================================

This script trains a YOLOv8 model on the pool balls detection dataset.
It includes comprehensive training configuration, validation, and result visualization.

Usage:
    python train_yolov8.py [--model-size MODEL_SIZE] [--epochs EPOCHS] [--batch-size BATCH_SIZE]

Example:
    python train_yolov8.py --model-size yolo8n --epochs 100 --batch-size 16
"""

import os
import argparse
import yaml
from pathlib import Path
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class YOLOv8Trainer:
    """YOLOv8 Training Class for Pool Balls Detection"""
    
    def __init__(self, data_yaml_path="data.yaml", model_size="yolo8n"):
        """
        Initialize the YOLOv8 trainer
        
        Args:
            data_yaml_path (str): Path to the data.yaml configuration file
            model_size (str): YOLOv8 model size (yolo8n, yolo8s, yolo8m, yolo8l, yolo8x)
        """
        self.data_yaml_path = data_yaml_path
        self.model_size = model_size
        self.model = None
        self.results = None
        
        # Verify data.yaml exists
        if not os.path.exists(data_yaml_path):
            raise FileNotFoundError(f"Data configuration file not found: {data_yaml_path}")
        
        # Load and verify dataset configuration
        self.load_dataset_config()
        
        # Create output directories
        self.create_output_directories()
        
    def load_dataset_config(self):
        """Load and validate dataset configuration"""
        with open(self.data_yaml_path, 'r') as file:
            self.data_config = yaml.safe_load(file)
        
        logger.info(f"Dataset configuration loaded:")
        logger.info(f"  - Number of classes: {self.data_config['nc']}")
        logger.info(f"  - Class names: {self.data_config['names']}")
        logger.info(f"  - Train path: {self.data_config['train']}")
        logger.info(f"  - Validation path: {self.data_config['val']}")
        logger.info(f"  - Test path: {self.data_config['test']}")
        
        # Verify dataset paths exist
        for split in ['train', 'val', 'test']:
            if split in self.data_config:
                path = self.data_config[split].replace('../', '')
                if not os.path.exists(path):
                    logger.warning(f"Warning: {split} path does not exist: {path}")
    
    def create_output_directories(self):
        """Create necessary output directories"""
        self.output_dir = Path("runs/train")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create timestamped run directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.output_dir / f"pool_balls_{self.model_size}_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Output directory created: {self.run_dir}")
    
    def initialize_model(self):
        """Initialize the YOLOv8 model"""
        try:
            self.model = YOLO(f"{self.model_size}.pt")
            logger.info(f"YOLOv8 model initialized: {self.model_size}")
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            raise
    
    def train(self, epochs=100, batch_size=16, imgsz=640, device=None, **kwargs):
        """
        Train the YOLOv8 model
        
        Args:
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            imgsz (int): Image size for training
            device (str): Device to use for training ('cpu', 'cuda', or None for auto)
            **kwargs: Additional training parameters
        """
        if self.model is None:
            self.initialize_model()
        
        # Set device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        logger.info(f"Starting training on device: {device}")
        logger.info(f"Training parameters:")
        logger.info(f"  - Epochs: {epochs}")
        logger.info(f"  - Batch size: {batch_size}")
        logger.info(f"  - Image size: {imgsz}")
        logger.info(f"  - Model: {self.model_size}")
        
        # Training parameters
        train_params = {
            'data': self.data_yaml_path,
            'epochs': epochs,
            'batch': batch_size,
            'imgsz': imgsz,
            'device': device,
            'project': str(self.run_dir.parent),
            'name': self.run_dir.name,
            'save': True,
            'save_period': 10,  # Save checkpoint every 10 epochs
            'cache': True,  # Cache images for faster training
            'workers': 8,  # Number of worker threads
            'patience': 50,  # Early stopping patience
            'lr0': 0.01,  # Initial learning rate
            'lrf': 0.01,  # Final learning rate
            'momentum': 0.937,  # SGD momentum
            'weight_decay': 0.0005,  # Optimizer weight decay
            'warmup_epochs': 3,  # Warmup epochs
            'warmup_momentum': 0.8,  # Warmup momentum
            'warmup_bias_lr': 0.1,  # Warmup bias learning rate
            'box': 7.5,  # Box loss gain
            'cls': 0.5,  # Classification loss gain
            'dfl': 1.5,  # DFL loss gain
            'pose': 12.0,  # Pose loss gain
            'kobj': 2.0,  # Keypoint object loss gain
            'label_smoothing': 0.0,  # Label smoothing
            'nbs': 64,  # Nominal batch size
            'overlap_mask': True,  # Overlap mask during training
            'mask_ratio': 4,  # Mask downsample ratio
            'dropout': 0.0,  # Use dropout regularization
            'val': True,  # Validate during training
            'plots': True,  # Generate training plots
            'verbose': True,  # Verbose output
            'seed': 0,  # Random seed for reproducibility
            'deterministic': True,  # Deterministic training
            'single_cls': False,  # Treat dataset as single-class
            'rect': False,  # Rectangular training
            'cos_lr': False,  # Cosine LR scheduler
            'close_mosaic': 10,  # Disable mosaic augmentation for final epochs
            'resume': False,  # Resume training from last checkpoint
            'amp': True,  # Automatic Mixed Precision (AMP) training
            'fraction': 1.0,  # Dataset fraction to train on
            'profile': False,  # Profile ONNX and TensorRT speeds during training
            'freeze': None,  # Freeze layers: backbone=10, first3=0,1,2
            'multi_scale': False,  # Multi-scale training
            'overlap_mask': True,  # Overlap mask during training
            'mask_ratio': 4,  # Mask downsample ratio
            'dropout': 0.0,  # Use dropout regularization
            'val': True,  # Validate during training
            'plots': True,  # Generate training plots
            'verbose': True,  # Verbose output
            'seed': 0,  # Random seed for reproducibility
            'deterministic': True,  # Deterministic training
            'single_cls': False,  # Treat dataset as single-class
            'rect': False,  # Rectangular training
            'cos_lr': False,  # Cosine LR scheduler
            'close_mosaic': 10,  # Disable mosaic augmentation for final epochs
            'resume': False,  # Resume training from last checkpoint
            'amp': True,  # Automatic Mixed Precision (AMP) training
            'fraction': 1.0,  # Dataset fraction to train on
            'profile': False,  # Profile ONNX and TensorRT speeds during training
            'freeze': None,  # Freeze layers: backbone=10, first3=0,1,2
            'multi_scale': False,  # Multi-scale training
        }
        
        # Update with any additional parameters
        train_params.update(kwargs)
        
        try:
            # Start training
            self.results = self.model.train(**train_params)
            logger.info("Training completed successfully!")
            
            # Save training summary
            self.save_training_summary()
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def validate(self, data_yaml_path=None):
        """
        Validate the trained model
        
        Args:
            data_yaml_path (str): Path to validation data (uses training data if None)
        """
        if self.model is None:
            logger.error("No model available for validation. Train the model first.")
            return None
        
        data_path = data_yaml_path or self.data_yaml_path
        
        logger.info("Starting model validation...")
        
        try:
            # Validate the model
            val_results = self.model.val(data=data_path)
            
            logger.info("Validation completed!")
            logger.info(f"mAP50: {val_results.box.map50:.4f}")
            logger.info(f"mAP50-95: {val_results.box.map:.4f}")
            
            return val_results
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            raise
    
    def test(self, test_data_path=None):
        """
        Test the trained model on test dataset
        
        Args:
            test_data_path (str): Path to test data (uses data.yaml test path if None)
        """
        if self.model is None:
            logger.error("No model available for testing. Train the model first.")
            return None
        
        # Use test data from data.yaml if not specified
        if test_data_path is None:
            test_data_path = self.data_config.get('test', 'test/images')
        
        logger.info(f"Testing model on: {test_data_path}")
        
        try:
            # Run inference on test images
            test_results = self.model(test_data_path)
            
            logger.info("Testing completed!")
            return test_results
            
        except Exception as e:
            logger.error(f"Testing failed: {e}")
            raise
    
    def save_training_summary(self):
        """Save training summary to file"""
        if self.results is None:
            logger.warning("No training results to save")
            return
        
        summary_file = self.run_dir / "training_summary.txt"
        
        with open(summary_file, 'w') as f:
            f.write("YOLOv8 Pool Balls Detection Training Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Model: {self.model_size}\n")
            f.write(f"Dataset: {self.data_yaml_path}\n")
            f.write(f"Classes: {self.data_config['nc']}\n")
            f.write(f"Class names: {', '.join(self.data_config['names'])}\n")
            f.write(f"Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            if hasattr(self.results, 'results_dict'):
                f.write("Training Results:\n")
                for key, value in self.results.results_dict.items():
                    f.write(f"  {key}: {value}\n")
        
        logger.info(f"Training summary saved to: {summary_file}")
    
    def plot_training_results(self):
        """Plot and save training results"""
        if self.results is None:
            logger.warning("No training results to plot")
            return
        
        try:
            # Create plots directory
            plots_dir = self.run_dir / "plots"
            plots_dir.mkdir(exist_ok=True)
            
            # Plot training curves
            results_dir = self.run_dir / "results.csv"
            if results_dir.exists():
                import pandas as pd
                df = pd.read_csv(results_dir)
                
                # Create figure with subplots
                fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                fig.suptitle('YOLOv8 Training Results', fontsize=16)
                
                # Plot loss curves
                if 'train/box_loss' in df.columns:
                    axes[0, 0].plot(df['epoch'], df['train/box_loss'], label='Train Box Loss')
                    axes[0, 0].plot(df['epoch'], df['val/box_loss'], label='Val Box Loss')
                    axes[0, 0].set_title('Box Loss')
                    axes[0, 0].set_xlabel('Epoch')
                    axes[0, 0].set_ylabel('Loss')
                    axes[0, 0].legend()
                    axes[0, 0].grid(True)
                
                # Plot classification loss
                if 'train/cls_loss' in df.columns:
                    axes[0, 1].plot(df['epoch'], df['train/cls_loss'], label='Train Cls Loss')
                    axes[0, 1].plot(df['epoch'], df['val/cls_loss'], label='Val Cls Loss')
                    axes[0, 1].set_title('Classification Loss')
                    axes[0, 1].set_xlabel('Epoch')
                    axes[0, 1].set_ylabel('Loss')
                    axes[0, 1].legend()
                    axes[0, 1].grid(True)
                
                # Plot mAP
                if 'metrics/mAP50(B)' in df.columns:
                    axes[1, 0].plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP@0.5')
                    axes[1, 0].plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP@0.5:0.95')
                    axes[1, 0].set_title('Mean Average Precision')
                    axes[1, 0].set_xlabel('Epoch')
                    axes[1, 0].set_ylabel('mAP')
                    axes[1, 0].legend()
                    axes[1, 0].grid(True)
                
                # Plot learning rate
                if 'lr/pg0' in df.columns:
                    axes[1, 1].plot(df['epoch'], df['lr/pg0'], label='Learning Rate')
                    axes[1, 1].set_title('Learning Rate')
                    axes[1, 1].set_xlabel('Epoch')
                    axes[1, 1].set_ylabel('LR')
                    axes[1, 1].legend()
                    axes[1, 1].grid(True)
                
                plt.tight_layout()
                plt.savefig(plots_dir / "training_curves.png", dpi=300, bbox_inches='tight')
                plt.close()
                
                logger.info(f"Training plots saved to: {plots_dir}")
            
        except Exception as e:
            logger.error(f"Failed to create training plots: {e}")


def main():
    """Main function to run the training script"""
    parser = argparse.ArgumentParser(description='Train YOLOv8 model for pool balls detection')
    parser.add_argument('--model-size', type=str, default='yolo8n',
                       choices=['yolo8n', 'yolo8s', 'yolo8m', 'yolo8l', 'yolo8x'],
                       help='YOLOv8 model size (default: yolo8n)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs (default: 100)')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for training (default: 16)')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Image size for training (default: 640)')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cpu, cuda, or auto)')
    parser.add_argument('--data-yaml', type=str, default='data.yaml',
                       help='Path to data.yaml file (default: data.yaml)')
    parser.add_argument('--validate', action='store_true',
                       help='Run validation after training')
    parser.add_argument('--test', action='store_true',
                       help='Run test after training')
    
    args = parser.parse_args()
    
    try:
        # Initialize trainer
        logger.info("Initializing YOLOv8 trainer...")
        trainer = YOLOv8Trainer(data_yaml_path=args.data_yaml, model_size=args.model_size)
        
        # Train the model
        logger.info("Starting training...")
        trainer.train(
            epochs=args.epochs,
            batch_size=args.batch_size,
            imgsz=args.imgsz,
            device=args.device
        )
        
        # Plot training results
        trainer.plot_training_results()
        
        # Run validation if requested
        if args.validate:
            logger.info("Running validation...")
            trainer.validate()
        
        # Run test if requested
        if args.test:
            logger.info("Running test...")
            trainer.test()
        
        logger.info("Training pipeline completed successfully!")
        logger.info(f"Results saved to: {trainer.run_dir}")
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()
