import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from torch.utils.data import DataLoader

from model import TabularTransformerWithAttention
from dataset import MultiplicationTableDataset, collate_fn


def visualize_attention(
    model,
    dataset,
    sample_idx=0,
    layer_idx=0,
    save_path=None,
    device='mps'
):
    """
    Visualize attention weights for a specific sample.

    This shows which cells the model attends to when predicting the masked cell.
    For your financial forecasting: this is like seeing which balance sheet items
    the model looks at when predicting a future value.

    Args:
        model: Trained model
        dataset: Dataset to sample from
        sample_idx: Which sample to visualize
        layer_idx: Which transformer layer to visualize
        save_path: Path to save the visualization
        device: Device to run on
    """
    model.eval()

    # Get a sample
    sample = dataset[sample_idx]
    row_ids = sample['row_ids'].unsqueeze(0).to(device)
    col_ids = sample['col_ids'].unsqueeze(0).to(device)
    values = sample['values'].unsqueeze(0).to(device)
    target_idx = sample['target_idx']
    target_value = sample['target_value'].item()
    target_row = sample['target_row']
    target_col = sample['target_col']

    # Forward pass
    with torch.no_grad():
        predictions, attention_weights = model(row_ids, col_ids, values)

    # Get prediction for masked cell
    predicted_value = predictions[0, target_idx, 0].item()

    # Get attention weights for specified layer
    attn = attention_weights[layer_idx][0].cpu().numpy()  # (seq_len, seq_len)

    # Create labels for cells
    labels = []
    for i in range(len(row_ids[0])):
        r = row_ids[0, i].item() + 1  # +1 for 1-indexed display
        c = col_ids[0, i].item() + 1
        v = values[0, i, 0].item()
        if i == target_idx:
            labels.append(f"[{r}×{c}]=?")
        else:
            labels.append(f"[{r}×{c}]={int(v)}")

    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Attention heatmap
    sns.heatmap(
        attn,
        annot=True,
        fmt='.2f',
        cmap='YlOrRd',
        xticklabels=labels,
        yticklabels=labels,
        ax=axes[0],
        cbar_kws={'label': 'Attention Weight'}
    )
    axes[0].set_title(f'Attention Weights (Layer {layer_idx})')
    axes[0].set_xlabel('Key (Attending From)')
    axes[0].set_ylabel('Query (Attending To)')

    # Plot 2: Attention to target cell
    target_attention = attn[target_idx, :]  # What the target cell attends to

    axes[1].bar(range(len(labels)), target_attention, color='steelblue')
    axes[1].set_xticks(range(len(labels)))
    axes[1].set_xticklabels(labels, rotation=45, ha='right')
    axes[1].set_ylabel('Attention Weight')
    axes[1].set_title(
        f'What the target cell [{target_row+1}×{target_col+1}] attends to\n'
        f'True: {target_value:.0f}, Predicted: {predicted_value:.2f}'
    )
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()

    return attn, predicted_value


def visualize_multiple_layers(
    model,
    dataset,
    sample_idx=0,
    save_path=None,
    device='mps'
):
    """
    Visualize attention across all layers for a single sample.

    This helps understand how attention patterns evolve through the network.
    Early layers might focus on local patterns, later layers on global relationships.
    """
    model.eval()

    # Get a sample
    sample = dataset[sample_idx]
    row_ids = sample['row_ids'].unsqueeze(0).to(device)
    col_ids = sample['col_ids'].unsqueeze(0).to(device)
    values = sample['values'].unsqueeze(0).to(device)
    target_idx = sample['target_idx']
    target_value = sample['target_value'].item()
    target_row = sample['target_row']
    target_col = sample['target_col']

    # Forward pass
    with torch.no_grad():
        predictions, attention_weights = model(row_ids, col_ids, values)

    predicted_value = predictions[0, target_idx, 0].item()

    num_layers = len(attention_weights)

    # Create labels
    labels = []
    for i in range(len(row_ids[0])):
        r = row_ids[0, i].item() + 1
        c = col_ids[0, i].item() + 1
        v = values[0, i, 0].item()
        if i == target_idx:
            labels.append(f"[{r}×{c}]=?")
        else:
            labels.append(f"[{r}×{c}]={int(v)}")

    # Create subplots for each layer
    fig, axes = plt.subplots(1, num_layers, figsize=(8 * num_layers, 6))

    if num_layers == 1:
        axes = [axes]

    for layer_idx in range(num_layers):
        attn = attention_weights[layer_idx][0].cpu().numpy()
        target_attention = attn[target_idx, :]

        axes[layer_idx].bar(range(len(labels)), target_attention, color='steelblue')
        axes[layer_idx].set_xticks(range(len(labels)))
        axes[layer_idx].set_xticklabels(labels, rotation=45, ha='right')
        axes[layer_idx].set_ylabel('Attention Weight')
        axes[layer_idx].set_title(f'Layer {layer_idx} Attention')
        axes[layer_idx].grid(axis='y', alpha=0.3)
        axes[layer_idx].set_ylim([0, max(1.0, target_attention.max() * 1.1)])

    fig.suptitle(
        f'Target: [{target_row+1}×{target_col+1}] = {target_value:.0f}, '
        f'Predicted: {predicted_value:.2f}',
        fontsize=14,
        y=1.02
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()


def analyze_attention_patterns(
    model,
    dataset,
    num_samples=10,
    save_path=None,
    device='mps'
):
    """
    Analyze attention patterns across multiple samples to find trends.

    For your use case: This could reveal which balance sheet items
    are consistently important for predictions.
    """
    model.eval()

    # Collect attention patterns
    all_attention_to_row = []
    all_attention_to_col = []

    for idx in range(min(num_samples, len(dataset))):
        sample = dataset[idx]
        row_ids = sample['row_ids'].unsqueeze(0).to(device)
        col_ids = sample['col_ids'].unsqueeze(0).to(device)
        values = sample['values'].unsqueeze(0).to(device)
        target_idx = sample['target_idx']
        target_row = sample['target_row']
        target_col = sample['target_col']

        with torch.no_grad():
            predictions, attention_weights = model(row_ids, col_ids, values)

        # Average attention across layers
        avg_attention = torch.stack(attention_weights).mean(dim=0)
        target_attention = avg_attention[0, target_idx, :].cpu().numpy()

        # Analyze: does it attend more to same row or same column?
        row_ids_np = row_ids[0].cpu().numpy()
        col_ids_np = col_ids[0].cpu().numpy()

        same_row_mask = row_ids_np == target_row
        same_col_mask = col_ids_np == target_col

        attention_to_same_row = target_attention[same_row_mask].sum()
        attention_to_same_col = target_attention[same_col_mask].sum()

        all_attention_to_row.append(attention_to_same_row)
        all_attention_to_col.append(attention_to_same_col)

    # Visualize results
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot attention distribution
    axes[0].scatter(all_attention_to_row, all_attention_to_col, alpha=0.6)
    axes[0].plot([0, 1], [0, 1], 'r--', label='Equal attention')
    axes[0].set_xlabel('Attention to Same Row')
    axes[0].set_ylabel('Attention to Same Column')
    axes[0].set_title('Attention Distribution Across Samples')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Plot histogram
    axes[1].hist(
        [all_attention_to_row, all_attention_to_col],
        label=['Same Row', 'Same Column'],
        bins=10,
        alpha=0.7
    )
    axes[1].set_xlabel('Total Attention Weight')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Attention Pattern Analysis')
    axes[1].legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved analysis to {save_path}")
    else:
        plt.show()

    # Print summary
    print("\n=== Attention Pattern Analysis ===")
    print(f"Avg attention to same row: {np.mean(all_attention_to_row):.3f} ± {np.std(all_attention_to_row):.3f}")
    print(f"Avg attention to same column: {np.mean(all_attention_to_col):.3f} ± {np.std(all_attention_to_col):.3f}")


if __name__ == "__main__":
    # Load trained model
    device = 'mps'

    # Create model
    model = TabularTransformerWithAttention(
        d_model=64,
        nhead=4,
        num_layers=2,
        max_rows=12,
        max_cols=12
    ).to(device)

    # Load checkpoint
    checkpoint = torch.load('checkpoints/best_model.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from epoch {checkpoint['epoch']} with val_loss {checkpoint['val_loss']:.4f}")

    # Create test dataset
    test_dataset = MultiplicationTableDataset(
        max_row=12, max_col=12, context_size=10, split='test'
    )

    # Visualize attention for a few samples
    import os
    os.makedirs('visualizations', exist_ok=True)

    for i in range(3):
        print(f"\n=== Visualizing sample {i} ===")
        visualize_attention(
            model, test_dataset, sample_idx=i, layer_idx=0,
            save_path=f'visualizations/attention_sample_{i}.png',
            device=device
        )

        visualize_multiple_layers(
            model, test_dataset, sample_idx=i,
            save_path=f'visualizations/layers_sample_{i}.png',
            device=device
        )

    # Analyze patterns across multiple samples
    analyze_attention_patterns(
        model, test_dataset, num_samples=20,
        save_path='visualizations/attention_analysis.png',
        device=device
    )
