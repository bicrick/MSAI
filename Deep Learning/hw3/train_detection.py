import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

from homework.datasets.road_dataset import load_data
from homework.metrics import DetectionMetric
from homework.models import Detector, save_model


def focal_loss(probs, targets, gamma=2.0, alpha=0.25):
    """
    Compute focal loss for multi-class segmentation
    
    Args:
        probs: (B, C, H, W) softmax probabilities
        targets: (B, H, W) target class indices
        gamma: focusing parameter
        alpha: weighting factor
    
    Returns:
        focal_loss: scalar focal loss
    """
    # Convert targets to one-hot encoding
    num_classes = probs.size(1)
    batch_size = probs.size(0)
    one_hot = torch.zeros_like(probs).to(probs.device)
    one_hot.scatter_(1, targets.unsqueeze(1), 1)
    
    # Compute focal weight
    pt = (one_hot * probs).sum(1)  # Get the probability of the correct class
    focal_weight = (1 - pt).pow(gamma)
    
    # Compute log probs and apply focal weight
    log_probs = F.log_softmax(probs, dim=1)
    loss = -focal_weight.unsqueeze(1) * one_hot * log_probs
    
    # Apply alpha weighting to balance classes
    alpha_weight = torch.ones_like(one_hot) * (1 - alpha)
    alpha_weight[:, 1:] = alpha  # Higher weight for non-background classes
    
    loss = loss * alpha_weight
    
    # Average over batch, height, width
    return loss.sum() / (batch_size * probs.size(2) * probs.size(3))


def train(args):
    # Set device optimized for Apple Silicon if available
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"Using MPS (Apple Silicon GPU) acceleration")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA GPU acceleration")
    else:
        device = torch.device("cpu")
        print(f"Using CPU")

    # Set up mixed precision training if supported
    # Note: As of this implementation, MPS doesn't fully support AMP
    use_amp = device.type == 'cuda'  # Only use AMP with CUDA for now
    scaler = GradScaler() if use_amp else None
    if use_amp:
        print("Using mixed precision training")
    else:
        print("Mixed precision not available, using full precision")

    # Create directories for logs and checkpoints
    log_dir = Path(args.log_dir)
    log_dir.mkdir(exist_ok=True, parents=True)
    
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    
    writer = SummaryWriter(log_dir=log_dir)

    # Load datasets
    train_loader = load_data(
        args.train_path,
        transform_pipeline="default",  # Use default transforms for now
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
    )
    val_loader = load_data(
        args.val_path,
        transform_pipeline="default",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )
    print(f"Training dataset size: {len(train_loader.dataset)}")
    print(f"Validation dataset size: {len(val_loader.dataset)}")

    # Create model
    model = Detector(in_channels=3, num_classes=3)
    model.to(device)

    # Define loss functions with weights for segmentation
    # Use weighted cross-entropy loss for segmentation to address class imbalance
    # Weight classes: Background (class 0) much lower, Lane boundaries (class 1, 2) much higher
    seg_weights = torch.tensor([0.05, 2.0, 2.0], device=device)  # Weights for background, left lane, right lane
    seg_criterion = nn.CrossEntropyLoss(weight=seg_weights)
    
    # L1 Loss (MAE) for depth regression
    depth_criterion = nn.L1Loss()
    
    # Define optimizer with weight decay for regularization
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )

    # Initialize training variables
    start_epoch = 0
    best_val_iou = 0.0
    
    # Initialize early stopping variables
    early_stopping_counter = 0
    early_stopping_best_iou = 0
    
    # Resume from checkpoint if specified
    if args.resume and os.path.isfile(args.resume):
        print(f"Loading checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_iou = checkpoint['best_val_iou']
        if scaler is not None and 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        print(f"Resuming training from epoch {start_epoch} with best IoU {best_val_iou:.4f}")

    # Initialize metrics
    train_metric = DetectionMetric(num_classes=3)
    val_metric = DetectionMetric(num_classes=3)

    # Training loop
    print(f"Starting training for {args.epochs} epochs")
    
    for epoch in range(start_epoch, args.epochs):
        # Training phase
        model.train()
        train_metric.reset()
        train_seg_loss = 0.0
        train_depth_loss = 0.0
        train_total_loss = 0.0
        
        # Track epoch time
        epoch_start_time = time.time()
        batch_times = []

        for batch_idx, batch in enumerate(train_loader):
            batch_start_time = time.time()
            
            images = batch["image"].to(device)
            seg_labels = batch["track"].to(device)
            depth_labels = batch["depth"].to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass with mixed precision where supported
            if use_amp:
                with autocast():
                    seg_logits, depth_preds = model(images)
                    
                    # Calculate losses
                    seg_loss = seg_criterion(seg_logits, seg_labels)
                    
                    # Add focal loss component 
                    seg_probs = F.softmax(seg_logits, dim=1)
                    focal = focal_loss(seg_probs, seg_labels, gamma=2.0, alpha=0.75)
                    seg_loss = seg_loss + focal
                    
                    depth_loss = depth_criterion(depth_preds, depth_labels)
                    
                    # Calculate additional IoU loss
                    iou_loss = 0.0
                    for cls in range(1, 3):  # Only for lane classes (1, 2)
                        pred_mask = seg_probs[:, cls]
                        true_mask = (seg_labels == cls).float()
                        
                        smooth = 1e-6
                        intersection = (pred_mask * true_mask).sum(dim=(1, 2))
                        union = pred_mask.sum(dim=(1, 2)) + true_mask.sum(dim=(1, 2)) - intersection + smooth
                        
                        sample_iou = intersection / union
                        batch_iou_loss = (1.0 - sample_iou).mean()
                        iou_loss += batch_iou_loss
                    
                    # Weighted combination of losses
                    total_loss = args.seg_weight * seg_loss + args.depth_weight * depth_loss + args.iou_weight * iou_loss
                
                # Backward pass with gradient scaling
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard precision training
                seg_logits, depth_preds = model(images)
                
                # Calculate losses
                seg_loss = seg_criterion(seg_logits, seg_labels)
                
                # Add focal loss component 
                seg_probs = F.softmax(seg_logits, dim=1)
                focal = focal_loss(seg_probs, seg_labels, gamma=2.0, alpha=0.75)
                seg_loss = seg_loss + focal
                
                depth_loss = depth_criterion(depth_preds, depth_labels)
                
                # Calculate additional IoU loss for segmentation to directly optimize IoU
                iou_loss = 0.0
                # For each class
                for cls in range(1, 3):  # Only for lane classes (1, 2)
                    # Calculate intersection and union
                    pred_mask = seg_probs[:, cls]
                    true_mask = (seg_labels == cls).float()
                    
                    # Add small epsilon to avoid division by zero
                    smooth = 1e-6
                    
                    # Calculate intersection and union
                    intersection = (pred_mask * true_mask).sum(dim=(1, 2))
                    union = pred_mask.sum(dim=(1, 2)) + true_mask.sum(dim=(1, 2)) - intersection + smooth
                    
                    # Calculate IoU for each sample in batch
                    sample_iou = intersection / union
                    
                    # Calculate batch IoU loss (1 - IoU to minimize)
                    batch_iou_loss = (1.0 - sample_iou).mean()
                    iou_loss += batch_iou_loss
                
                # Weighted combination of losses
                total_loss = args.seg_weight * seg_loss + args.depth_weight * depth_loss + args.iou_weight * iou_loss
                
                # Backward pass and optimize
                total_loss.backward()
                optimizer.step()
            
            # Update metrics
            with torch.no_grad():
                preds, depths = model.predict(images)
                train_metric.add(preds, seg_labels, depths, depth_labels)
                
            # Accumulate losses
            train_seg_loss += seg_loss.item()
            train_depth_loss += depth_loss.item()
            train_total_loss += total_loss.item()
            
            # Track batch processing time
            batch_end_time = time.time()
            batch_time = batch_end_time - batch_start_time
            batch_times.append(batch_time)
            
            # Print progress
            if (batch_idx + 1) % 20 == 0:
                avg_batch_time = sum(batch_times[-20:]) / min(20, len(batch_times[-20:]))
                images_per_sec = args.batch_size / avg_batch_time
                print(f"Epoch [{epoch+1}/{args.epochs}], Batch [{batch_idx+1}/{len(train_loader)}], "
                      f"Seg Loss: {seg_loss.item():.4f}, Depth Loss: {depth_loss.item():.4f}, "
                      f"Total Loss: {total_loss.item():.4f}, "
                      f"Speed: {images_per_sec:.1f} img/s")
        
        # Calculate average training metrics
        avg_train_seg_loss = train_seg_loss / len(train_loader)
        avg_train_depth_loss = train_depth_loss / len(train_loader)
        avg_train_total_loss = train_total_loss / len(train_loader)
        train_metrics = train_metric.compute()
        
        # Log training metrics
        writer.add_scalar("Loss/train/segmentation", avg_train_seg_loss, epoch)
        writer.add_scalar("Loss/train/depth", avg_train_depth_loss, epoch)
        writer.add_scalar("Loss/train/total", avg_train_total_loss, epoch)
        writer.add_scalar("Metrics/train/iou", train_metrics["iou"], epoch)
        writer.add_scalar("Metrics/train/accuracy", train_metrics["accuracy"], epoch)
        writer.add_scalar("Metrics/train/abs_depth_error", train_metrics["abs_depth_error"], epoch)
        writer.add_scalar("Metrics/train/tp_depth_error", train_metrics["tp_depth_error"], epoch)
        
        # Validation phase
        model.eval()
        val_metric.reset()
        val_seg_loss = 0.0
        val_depth_loss = 0.0
        val_total_loss = 0.0
        
        with torch.inference_mode():
            for batch in val_loader:
                images = batch["image"].to(device)
                seg_labels = batch["track"].to(device)
                depth_labels = batch["depth"].to(device)
                
                # Forward pass
                seg_logits, depth_preds = model(images)
                
                # Calculate losses
                seg_loss = seg_criterion(seg_logits, seg_labels)

                # Add focal loss component 
                seg_probs = F.softmax(seg_logits, dim=1)
                focal = focal_loss(seg_probs, seg_labels, gamma=2.0, alpha=0.75)
                seg_loss = seg_loss + focal

                depth_loss = depth_criterion(depth_preds, depth_labels)
                
                # Calculate additional IoU loss for segmentation to directly optimize IoU
                iou_loss = 0.0
                # For each class
                for cls in range(1, 3):  # Only for lane classes (1, 2)
                    # Calculate intersection and union
                    pred_mask = seg_probs[:, cls]
                    true_mask = (seg_labels == cls).float()
                    
                    # Add small epsilon to avoid division by zero
                    smooth = 1e-6
                    
                    # Calculate intersection and union
                    intersection = (pred_mask * true_mask).sum(dim=(1, 2))
                    union = pred_mask.sum(dim=(1, 2)) + true_mask.sum(dim=(1, 2)) - intersection + smooth
                    
                    # Calculate IoU for each sample in batch
                    sample_iou = intersection / union
                    
                    # Calculate batch IoU loss (1 - IoU to minimize)
                    batch_iou_loss = (1.0 - sample_iou).mean()
                    iou_loss += batch_iou_loss

                # Weighted combination of losses
                total_loss = args.seg_weight * seg_loss + args.depth_weight * depth_loss + args.iou_weight * iou_loss
                
                # Update metrics
                preds, depths = model.predict(images)
                val_metric.add(preds, seg_labels, depths, depth_labels)
                
                # Accumulate losses
                val_seg_loss += seg_loss.item()
                val_depth_loss += depth_loss.item()
                val_total_loss += total_loss.item()
        
        # Calculate average validation metrics
        avg_val_seg_loss = val_seg_loss / len(val_loader)
        avg_val_depth_loss = val_depth_loss / len(val_loader)
        avg_val_total_loss = val_total_loss / len(val_loader)
        val_metrics = val_metric.compute()
        val_iou = val_metrics["iou"]
        
        # Log validation metrics
        writer.add_scalar("Loss/val/segmentation", avg_val_seg_loss, epoch)
        writer.add_scalar("Loss/val/depth", avg_val_depth_loss, epoch)
        writer.add_scalar("Loss/val/total", avg_val_total_loss, epoch)
        writer.add_scalar("Metrics/val/iou", val_metrics["iou"], epoch)
        writer.add_scalar("Metrics/val/accuracy", val_metrics["accuracy"], epoch)
        writer.add_scalar("Metrics/val/abs_depth_error", val_metrics["abs_depth_error"], epoch)
        writer.add_scalar("Metrics/val/tp_depth_error", val_metrics["tp_depth_error"], epoch)
        
        # Update learning rate scheduler based on IoU
        scheduler.step(val_iou)
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        avg_batch_time = sum(batch_times) / len(batch_times)
        images_per_sec = args.batch_size / avg_batch_time
        est_time_remaining = epoch_time * (args.epochs - epoch - 1)
        hours, remainder = divmod(est_time_remaining, 3600)
        minutes, seconds = divmod(remainder, 60)

        # Print epoch summary
        print(f"Epoch [{epoch+1}/{args.epochs}], "
              f"Train - Seg Loss: {avg_train_seg_loss:.4f}, Depth Loss: {avg_train_depth_loss:.4f}, "
              f"IOU: {train_metrics['iou']:.4f}, Depth Err: {train_metrics['abs_depth_error']:.4f}, "
              f"Val - Seg Loss: {avg_val_seg_loss:.4f}, Depth Loss: {avg_val_depth_loss:.4f}, "
              f"IOU: {val_iou:.4f}, Depth Err: {val_metrics['abs_depth_error']:.4f}")
        print(f"Epoch time: {epoch_time:.2f}s, Speed: {images_per_sec:.1f} img/s, Est. time remaining: {int(hours)}h {int(minutes)}m {int(seconds)}s")
        
        # Save best model based on validation IoU
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            print(f"New best validation IoU: {best_val_iou:.4f}")
            model_path = save_model(model)
            print(f"Model saved to {model_path}")

        # Save checkpoint every save_frequency epochs or on last epoch
        if (epoch + 1) % args.save_frequency == 0 or (epoch + 1) == args.epochs:
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pt"
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_iou': best_val_iou,
            }
            if scaler is not None:
                checkpoint['scaler_state_dict'] = scaler.state_dict()
            
            torch.save(checkpoint, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
        
        # Check for early stopping
        if val_iou < early_stopping_best_iou:
            early_stopping_counter += 1
            if early_stopping_counter >= args.early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
        else:
            early_stopping_counter = 0
            early_stopping_best_iou = val_iou
    
    # Final summary
    print(f"Training complete! Best validation IoU: {best_val_iou:.4f}")
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script for the road detection task")
    
    # Dataset paths
    parser.add_argument("--train_path", type=str, default="drive_data/train",
                        help="Path to the training dataset")
    parser.add_argument("--val_path", type=str, default="drive_data/val",
                        help="Path to the validation dataset")
    
    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for regularization")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    
    # Loss weights
    parser.add_argument("--seg_weight", type=float, default=1.0, help="Weight for segmentation loss")
    parser.add_argument("--depth_weight", type=float, default=5.0, help="Weight for depth loss")
    parser.add_argument("--iou_weight", type=float, default=3.0, help="Weight for IoU loss")
    
    # Misc
    parser.add_argument("--log_dir", type=str, default="logs/detection", 
                        help="Directory for TensorBoard logs")
    parser.add_argument("--num_workers", type=int, default=8, 
                        help="Number of data loading workers")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/detection", 
                        help="Directory for model checkpoints")
    parser.add_argument("--resume", type=str, default=None, 
                        help="Path to checkpoint to resume training from")
    parser.add_argument("--save_frequency", type=int, default=10, help="Frequency to save checkpoints")
    parser.add_argument("--early_stopping_patience", type=int, default=5, help="Patience for early stopping")
    
    args = parser.parse_args()
    train(args) 