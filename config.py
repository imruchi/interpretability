"""
Configuration file for training hyperparameters.
Modify these values to experiment with different model architectures.
"""

# Data Configuration
MAX_ROW = 12  # Size of multiplication table (12x12)
MAX_COL = 12
CONTEXT_SIZE = 10  # Number of cells provided as context

# Model Architecture
D_MODEL = 64  # Embedding dimension
NHEAD = 4  # Number of attention heads
NUM_LAYERS = 2  # Number of transformer layers
DIM_FEEDFORWARD = 256  # Hidden dimension in feedforward network
DROPOUT = 0.1  # Dropout rate

# Training Configuration
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
DEVICE = 'mps'  # 'mps' for Mac, 'cuda' for NVIDIA GPU, 'cpu' for CPU

# Paths
CHECKPOINT_DIR = 'checkpoints'
VISUALIZATION_DIR = 'visualizations'
