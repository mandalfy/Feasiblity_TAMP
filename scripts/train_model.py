#!/usr/bin/env python3
"""
Train feasibility classifier.

Usage:
    python scripts/train_model.py --model mlp --epochs 50
    python scripts/train_model.py --model cnn --epochs 30 --batch_size 32
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from data_generation.dataset import FeasibilityDataset, create_dataloaders
from models.mlp_classifier import MLPClassifier
from models.cnn_classifier import CNNClassifier
from training.train import Trainer
from training.evaluate import evaluate_model, print_evaluation_report, plot_evaluation


def main():
    parser = argparse.ArgumentParser(description="Train feasibility classifier")
    parser.add_argument(
        "--model",
        type=str,
        choices=["mlp", "cnn"],
        default="mlp",
        help="Model type"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/feasibility_dataset.h5",
        help="Path to dataset"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate"
    )
    parser.add_argument(
        "--hidden_dims",
        type=int,
        nargs="+",
        default=[256, 256, 128],
        help="Hidden layer dimensions for MLP"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints/",
        help="Directory for checkpoints"
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs/",
        help="Directory for TensorBoard logs"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device (auto, cuda, cpu)"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("TRAINING FEASIBILITY CLASSIFIER")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.data}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print("="*60)
    
    # Check dataset exists
    if not os.path.exists(args.data):
        print(f"Error: Dataset not found at {args.data}")
        print("Run 'python scripts/generate_data.py' first.")
        sys.exit(1)
        
    use_images = (args.model == "cnn")
    
    # Create data loaders
    train_loader, val_loader = create_dataloaders(
        args.data,
        batch_size=args.batch_size,
        use_images=use_images,
        num_workers=4
    )
    
    # Get dimensions
    dataset = FeasibilityDataset(args.data)
    state_dim, action_dim, input_dim = dataset.get_input_dims()
    print(f"State dim: {state_dim}, Action dim: {action_dim}")
    
    # Create model
    if args.model == "mlp":
        model = MLPClassifier(
            input_dim=input_dim,
            hidden_dims=args.hidden_dims,
            dropout=0.2
        )
    else:  # cnn
        model = CNNClassifier(
            action_dim=action_dim,
            backbone="resnet18",
            pretrained=True
        )
        
    print(f"Model parameters: {model.get_num_parameters():,}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=args.lr,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        use_images=use_images,
        device=args.device
    )
    
    # Train
    history = trainer.train(
        num_epochs=args.epochs,
        early_stopping_patience=10
    )
    
    # Evaluate
    print("\nEvaluating best model...")
    trainer.load_checkpoint(os.path.join(args.checkpoint_dir, "best_model.pt"))
    
    metrics = evaluate_model(
        model,
        val_loader,
        device=args.device,
        use_images=use_images
    )
    
    print_evaluation_report(metrics)
    plot_evaluation(metrics, save_dir="plots/")
    
    print("\nTraining complete!")
    print(f"Best model saved to: {args.checkpoint_dir}/best_model.pt")
    print(f"TensorBoard logs: tensorboard --logdir {args.log_dir}")


if __name__ == "__main__":
    main()
