import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from homework.datasets.classification_dataset import load_data
from homework.metrics import AccuracyMetric
from homework.models import Classifier, save_model


def train(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create directories for logs and checkpoints
    log_dir = Path(args.log_dir)
    log_dir.mkdir(exist_ok=True, parents=True)
    writer = SummaryWriter(log_dir=log_dir)

    # Load datasets
    train_loader = load_data(
        args.train_path,
        transform_pipeline="aug",  # Use data augmentation for training
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
    )
    val_loader = load_data(
        args.val_path,
        transform_pipeline="default",  # No augmentation for validation
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )
    print(f"Training dataset size: {len(train_loader.dataset)}")
    print(f"Validation dataset size: {len(val_loader.dataset)}")

    # Create model
    model = Classifier(in_channels=3, num_classes=6)
    model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True
    )

    # Initialize metrics
    train_metric = AccuracyMetric()
    val_metric = AccuracyMetric()

    # Training loop
    best_val_acc = 0.0
    print(f"Starting training for {args.epochs} epochs")
    
    for epoch in range(args.epochs):
        # Training phase
        model.train()
        train_metric.reset()
        train_loss = 0.0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update metrics
            preds = model.predict(images)
            train_metric.add(preds, labels)
            train_loss += loss.item()
            
            # Print progress
            if (batch_idx + 1) % 50 == 0:
                print(f"Epoch [{epoch+1}/{args.epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
        
        # Calculate average training metrics
        avg_train_loss = train_loss / len(train_loader)
        train_metrics = train_metric.compute()
        train_acc = train_metrics["accuracy"]
        
        # Log training metrics
        writer.add_scalar("Loss/train", avg_train_loss, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)
        
        # Validation phase
        model.eval()
        val_metric.reset()
        val_loss = 0.0
        
        with torch.inference_mode():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                
                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                # Update metrics
                preds = model.predict(images)
                val_metric.add(preds, labels)
                val_loss += loss.item()
        
        # Calculate average validation metrics
        avg_val_loss = val_loss / len(val_loader)
        val_metrics = val_metric.compute()
        val_acc = val_metrics["accuracy"]
        
        # Log validation metrics
        writer.add_scalar("Loss/val", avg_val_loss, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)
        
        # Update learning rate scheduler
        scheduler.step(val_acc)
        
        # Print epoch summary
        print(f"Epoch [{epoch+1}/{args.epochs}], "
              f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"New best validation accuracy: {best_val_acc:.4f}")
            model_path = save_model(model)
            print(f"Model saved to {model_path}")
    
    # Final summary
    print(f"Training complete! Best validation accuracy: {best_val_acc:.4f}")
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script for the classification task")
    
    # Dataset paths
    parser.add_argument("--train_path", type=str, default="classification_data/train",
                        help="Path to the training dataset")
    parser.add_argument("--val_path", type=str, default="classification_data/val",
                        help="Path to the validation dataset")
    
    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for regularization")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    
    # Misc
    parser.add_argument("--log_dir", type=str, default="logs/classification", 
                        help="Directory for TensorBoard logs")
    parser.add_argument("--num_workers", type=int, default=4, 
                        help="Number of data loading workers")
    
    args = parser.parse_args()
    train(args) 