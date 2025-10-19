"""
Training script for tabular transformer on multiplication table task.

This demonstrates mechanistic interpretability by training a transformer
to predict masked cells in a multiplication table, which mirrors the task
of predicting future balance sheet values in financial forecasting.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os
import argparse

from model import TabularTransformerWithAttention
from dataset import MultiplicationTableDataset, collate_fn
import config


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in tqdm(dataloader, desc="Training"):
        row_ids = batch['row_ids'].to(device)
        col_ids = batch['col_ids'].to(device)
        values = batch['values'].to(device)
        target_idx = batch['target_idx'].to(device)
        target_value = batch['target_value'].to(device)

        optimizer.zero_grad()

        # Forward pass
        predictions, attention_weights = model(row_ids, col_ids, values)

        # Get predictions for masked positions
        batch_size = predictions.size(0)
        batch_indices = torch.arange(batch_size, device=device)
        masked_predictions = predictions[batch_indices, target_idx].squeeze(-1)

        # Compute loss
        loss = criterion(masked_predictions, target_value)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Compute accuracy (within 10% tolerance)
        pred_rounded = masked_predictions.detach()
        tolerance = 0.1 * target_value.abs()
        correct += ((pred_rounded - target_value).abs() <= tolerance).sum().item()
        total += batch_size

    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total

    return avg_loss, accuracy


def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            row_ids = batch['row_ids'].to(device)
            col_ids = batch['col_ids'].to(device)
            values = batch['values'].to(device)
            target_idx = batch['target_idx'].to(device)
            target_value = batch['target_value'].to(device)

            # Forward pass
            predictions, _ = model(row_ids, col_ids, values)

            # Get predictions for masked positions
            batch_size = predictions.size(0)
            batch_indices = torch.arange(batch_size, device=device)
            masked_predictions = predictions[batch_indices, target_idx].squeeze(-1)

            # Compute loss
            loss = criterion(masked_predictions, target_value)
            total_loss += loss.item()

            # Compute accuracy
            pred_rounded = masked_predictions
            tolerance = 0.1 * target_value.abs()
            correct += ((pred_rounded - target_value).abs() <= tolerance).sum().item()
            total += batch_size

    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total

    return avg_loss, accuracy


def train_model(
    max_row=12,
    max_col=12,
    context_size=10,
    d_model=64,
    nhead=4,
    num_layers=2,
    dim_feedforward=256,
    batch_size=32,
    num_epochs=50,
    lr=0.001,
    device='mps'
):
    """Main training loop"""

    print(f"Using device: {device}")

    # Create datasets
    train_dataset = MultiplicationTableDataset(
        max_row=max_row, max_col=max_col, context_size=context_size, split='train'
    )
    val_dataset = MultiplicationTableDataset(
        max_row=max_row, max_col=max_col, context_size=context_size, split='val'
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    # Create model
    model = TabularTransformerWithAttention(
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        max_rows=max_row,
        max_cols=max_col
    ).to(device)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # Training loop
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs('checkpoints', exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, 'checkpoints/best_model.pt')
            print(f"Saved best model with val_loss: {val_loss:.4f}")

    print("\nTraining completed!")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train tabular transformer for mechanistic interpretability')
    parser.add_argument('--epochs', type=int, default=config.NUM_EPOCHS, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=config.BATCH_SIZE, help='Batch size')
    parser.add_argument('--lr', type=float, default=config.LEARNING_RATE, help='Learning rate')
    parser.add_argument('--device', type=str, default=config.DEVICE, help='Device (mps/cuda/cpu)')
    args = parser.parse_args()

    print("=" * 60)
    print("Mechanistic Interpretability PoC: Multiplication Table")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Table size: {config.MAX_ROW}x{config.MAX_COL}")
    print(f"  Model: d_model={config.D_MODEL}, heads={config.NHEAD}, layers={config.NUM_LAYERS}")
    print(f"  Training: epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}")
    print(f"  Device: {args.device}")
    print("=" * 60)

    # Train the model
    model = train_model(
        max_row=config.MAX_ROW,
        max_col=config.MAX_COL,
        context_size=config.CONTEXT_SIZE,
        d_model=config.D_MODEL,
        nhead=config.NHEAD,
        num_layers=config.NUM_LAYERS,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        lr=args.lr,
        device=args.device
    )
