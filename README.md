# Mechanistic Interpretability PoC: Multiplication Table Prediction

This PoC demonstrates mechanistic interpretability concepts for financial forecasting using a simple multiplication table task.

## Concept

The multiplication table task mimics your financial forecasting scenario:
- **Multiplication table cells** = Balance sheet entries
- **Row/Column position** = Different financial metrics (revenue, assets, etc.)
- **Predicting masked cell** = Forecasting future balance sheet values
- **Attention weights** = Which historical entries the model considers important

## Project Structure

```
finance_poc/
├── model.py          # Transformer model with attention extraction
├── dataset.py        # Multiplication table dataset
├── train.py          # Training script
├── visualize.py      # Attention visualization utilities
├── config.py         # Configuration and hyperparameters
├── requirements.txt  # Python dependencies
└── README.md         # This file
```

## Setup

Install dependencies:
```bash
pip install torch numpy matplotlib seaborn tqdm
```

## Usage

### 1. Train the model

Basic training:
```bash
python train.py
```

Custom training with arguments:
```bash
python train.py --epochs 100 --batch-size 64 --lr 0.0005
```

Available arguments:
- `--epochs`: Number of training epochs (default: 50)
- `--batch-size`: Batch size (default: 32)
- `--lr`: Learning rate (default: 0.001)
- `--device`: Device to use - 'mps', 'cuda', or 'cpu' (default: 'mps')

This will:
- Create a 12×12 multiplication table dataset
- Train a 2-layer Transformer to predict masked cells
- Save the best model to `checkpoints/best_model.pt`

### 2. Visualize attention patterns

```bash
python visualize.py
```

This generates visualizations in `visualizations/`:
- **Attention heatmaps**: Shows which cells attend to which
- **Layer-wise attention**: How patterns evolve through layers
- **Pattern analysis**: Statistical analysis of attention trends

## Key Insights for Financial Forecasting

### What the visualizations show:

1. **Attention Heatmap**:
   - Darker colors = stronger attention
   - Shows which balance sheet items influence predictions
   - For multiplication: model learns to attend to same row/column

2. **Layer Evolution**:
   - Early layers: focus on local patterns
   - Later layers: capture global relationships
   - For finance: early layers might look at individual metrics, later layers at financial ratios

3. **Pattern Analysis**:
   - Identifies consistent attention patterns
   - For finance: reveals which metrics are consistently important (e.g., revenue for profit predictions)

### Extending to Financial Forecasting:

1. **Replace multiplication table** with historical balance sheets
2. **Rows** = different companies or time periods
3. **Columns** = financial metrics (revenue, assets, liabilities, etc.)
4. **Mask future values** and predict them
5. **Analyze attention** to see which historical metrics matter most

### Manipulating Predictions:

From the visualizations, you can:
1. Identify which cells have high attention weights
2. Modify those weights to test causality
3. Observe how predictions change
4. This reveals which factors drive the model's forecasts

## Model Architecture

- **Input**: (row_id, col_id, value) tuples for each cell
- **Embeddings**: Separate embeddings for row and column positions
- **Transformer**: 2 layers, 4 attention heads per layer
- **Output**: Predicted value for masked cell

## Next Steps for Your Project

1. **Adapt the dataset**: Replace `MultiplicationTableDataset` with financial data
2. **Add temporal dimension**: Incorporate time-series patterns
3. **Multi-head analysis**: Visualize different attention heads separately
4. **Intervention experiments**: Modify attention weights and observe prediction changes
5. **Feature importance**: Rank financial metrics by attention scores
