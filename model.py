import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for row and column positions"""
    def __init__(self, d_model, max_len=100):
        super().__init__()
        self.encoding = nn.Embedding(max_len, d_model)

    def forward(self, positions):
        return self.encoding(positions)


class TabularTransformer(nn.Module):
    """
    Transformer for tabular data prediction.
    Each cell is represented as (row_id, col_id, value).
    """
    def __init__(
        self,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        max_rows=20,
        max_cols=20,
        dropout=0.1
    ):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead

        # Embeddings for row and column positions
        self.row_embedding = PositionalEncoding(d_model // 2, max_rows)
        self.col_embedding = PositionalEncoding(d_model // 2, max_cols)

        # Value projection
        self.value_projection = nn.Linear(1, d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection
        self.output_projection = nn.Linear(d_model, 1)

        # Store attention weights for visualization
        self.attention_weights = []

    def forward(self, row_ids, col_ids, values, mask_positions=None):
        """
        Args:
            row_ids: (batch, seq_len) - row indices for each cell
            col_ids: (batch, seq_len) - column indices for each cell
            values: (batch, seq_len, 1) - cell values
            mask_positions: (batch, seq_len) - boolean mask for padding

        Returns:
            predictions: (batch, seq_len, 1) - predicted values
        """
        batch_size, seq_len = row_ids.shape

        # Get embeddings
        row_emb = self.row_embedding(row_ids)  # (batch, seq_len, d_model//2)
        col_emb = self.col_embedding(col_ids)  # (batch, seq_len, d_model//2)
        val_emb = self.value_projection(values)  # (batch, seq_len, d_model)

        # Combine position embeddings
        pos_emb = torch.cat([row_emb, col_emb], dim=-1)  # (batch, seq_len, d_model)

        # Combine with value embeddings
        x = pos_emb + val_emb  # (batch, seq_len, d_model)

        # Clear previous attention weights
        self.attention_weights = []

        # Register hooks to capture attention weights
        hooks = []
        for layer in self.transformer.layers:
            hook = layer.self_attn.register_forward_hook(self._get_attention_hook())
            hooks.append(hook)

        # Pass through transformer
        output = self.transformer(x, src_key_padding_mask=mask_positions)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        # Project to value
        predictions = self.output_projection(output)

        return predictions

    def _get_attention_hook(self):
        """Hook to capture attention weights during forward pass"""
        def hook(module, input, output):
            # output[1] contains attention weights when return_attention_weights=True
            # However, standard nn.MultiheadAttention doesn't return this by default
            # We'll store the attention weights if available
            if len(output) > 1 and output[1] is not None:
                self.attention_weights.append(output[1].detach())
        return hook

    def get_attention_weights(self):
        """Return stored attention weights from last forward pass"""
        return self.attention_weights


class TabularTransformerWithAttention(nn.Module):
    """
    Modified version that explicitly captures attention weights.
    We'll use a custom transformer implementation for better attention extraction.
    """
    def __init__(
        self,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        max_rows=20,
        max_cols=20,
        dropout=0.1
    ):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers

        # Embeddings
        self.row_embedding = PositionalEncoding(d_model // 2, max_rows)
        self.col_embedding = PositionalEncoding(d_model // 2, max_cols)
        self.value_projection = nn.Linear(1, d_model)

        # Custom transformer layers to extract attention
        self.layers = nn.ModuleList([
            TransformerEncoderLayerWithAttention(
                d_model, nhead, dim_feedforward, dropout
            )
            for _ in range(num_layers)
        ])

        # Output projection
        self.output_projection = nn.Linear(d_model, 1)

    def forward(self, row_ids, col_ids, values, mask_positions=None):
        """
        Args:
            row_ids: (batch, seq_len) - row indices
            col_ids: (batch, seq_len) - column indices
            values: (batch, seq_len, 1) - cell values
            mask_positions: (batch, seq_len) - padding mask

        Returns:
            predictions: (batch, seq_len, 1)
            attention_weights: list of (batch, nhead, seq_len, seq_len) for each layer
        """
        # Get embeddings
        row_emb = self.row_embedding(row_ids)
        col_emb = self.col_embedding(col_ids)
        val_emb = self.value_projection(values)

        # Combine embeddings
        pos_emb = torch.cat([row_emb, col_emb], dim=-1)
        x = pos_emb + val_emb

        # Pass through layers and collect attention
        attention_weights = []
        for layer in self.layers:
            x, attn = layer(x, mask_positions)
            attention_weights.append(attn)

        # Project to output
        predictions = self.output_projection(x)

        return predictions, attention_weights


class TransformerEncoderLayerWithAttention(nn.Module):
    """Custom transformer layer that returns attention weights"""
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        # Self attention
        attn_output, attn_weights = self.self_attn(
            x, x, x,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=True  # Average across heads for visualization
        )
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)

        # Feedforward
        ff_output = self.linear2(self.dropout(torch.relu(self.linear1(x))))
        x = x + self.dropout2(ff_output)
        x = self.norm2(x)

        return x, attn_weights
