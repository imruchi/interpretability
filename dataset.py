import torch
from torch.utils.data import Dataset
import numpy as np


class MultiplicationTableDataset(Dataset):
    """
    Dataset for multiplication table prediction.
    Each sample contains cells from a multiplication table with one cell masked.

    Format: Given context cells (row, col, value), predict the masked cell value.
    This mimics: given historical balance sheet entries, predict a future entry.
    """

    def __init__(self, max_row=12, max_col=12, context_size=10, split='train'):
        """
        Args:
            max_row: Maximum row in multiplication table (e.g., 12 for 12x12 table)
            max_col: Maximum column in multiplication table
            context_size: Number of context cells to provide (including masked cell)
            split: 'train', 'val', or 'test'
        """
        self.max_row = max_row
        self.max_col = max_col
        self.context_size = context_size
        self.split = split

        # Create multiplication table
        self.table = self._create_table()

        # Generate samples
        self.samples = self._generate_samples()

    def _create_table(self):
        """Create the multiplication table"""
        rows = np.arange(1, self.max_row + 1)
        cols = np.arange(1, self.max_col + 1)
        table = np.outer(rows, cols)
        return table

    def _generate_samples(self):
        """
        Generate training samples by masking different cells.
        Each sample: select a target cell, provide context cells, mask target.
        """
        samples = []

        # For each cell in the table, create a sample
        for target_row in range(self.max_row):
            for target_col in range(self.max_col):
                # Select context cells (random selection)
                # In practice, for time-series like balance sheets,
                # you'd select historical entries
                sample = {
                    'target_row': target_row,
                    'target_col': target_col,
                    'target_value': self.table[target_row, target_col]
                }
                samples.append(sample)

        # Split into train/val/test
        np.random.seed(42)
        indices = np.random.permutation(len(samples))

        n_train = int(0.7 * len(samples))
        n_val = int(0.15 * len(samples))

        if self.split == 'train':
            samples = [samples[i] for i in indices[:n_train]]
        elif self.split == 'val':
            samples = [samples[i] for i in indices[n_train:n_train + n_val]]
        else:  # test
            samples = [samples[i] for i in indices[n_train + n_val:]]

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Return a sample with context cells and masked target.

        Returns:
            row_ids: (context_size,) - row index for each cell
            col_ids: (context_size,) - column index for each cell
            values: (context_size,) - cell values (target is masked as 0)
            target_idx: int - index of target cell in sequence
            target_value: float - actual value of target cell
        """
        sample = self.samples[idx]
        target_row = sample['target_row']
        target_col = sample['target_col']
        target_value = sample['target_value']

        # Select context cells
        # Strategy: include same row, same column, and random cells
        context_cells = []

        # Add cells from same row (helps model learn row patterns)
        for c in range(self.max_col):
            if c != target_col and len(context_cells) < self.context_size - 1:
                context_cells.append((target_row, c, self.table[target_row, c]))

        # Add cells from same column (helps model learn column patterns)
        for r in range(self.max_row):
            if r != target_row and len(context_cells) < self.context_size - 1:
                context_cells.append((r, target_col, self.table[r, target_col]))

        # Add random cells if needed
        while len(context_cells) < self.context_size - 1:
            r = np.random.randint(0, self.max_row)
            c = np.random.randint(0, self.max_col)
            if (r, c) != (target_row, target_col):
                context_cells.append((r, c, self.table[r, c]))

        # Shuffle context cells
        np.random.shuffle(context_cells)

        # Add target cell (masked) at a random position
        target_idx = np.random.randint(0, self.context_size)
        context_cells.insert(target_idx, (target_row, target_col, 0.0))  # masked value

        # Convert to tensors
        row_ids = torch.tensor([c[0] for c in context_cells], dtype=torch.long)
        col_ids = torch.tensor([c[1] for c in context_cells], dtype=torch.long)
        values = torch.tensor([c[2] for c in context_cells], dtype=torch.float32).unsqueeze(-1)

        return {
            'row_ids': row_ids,
            'col_ids': col_ids,
            'values': values,
            'target_idx': target_idx,
            'target_value': torch.tensor(target_value, dtype=torch.float32),
            'target_row': target_row,
            'target_col': target_col
        }


def collate_fn(batch):
    """Custom collate function for batching"""
    row_ids = torch.stack([item['row_ids'] for item in batch])
    col_ids = torch.stack([item['col_ids'] for item in batch])
    values = torch.stack([item['values'] for item in batch])
    target_idx = torch.tensor([item['target_idx'] for item in batch], dtype=torch.long)
    target_value = torch.stack([item['target_value'] for item in batch])
    target_row = torch.tensor([item['target_row'] for item in batch], dtype=torch.long)
    target_col = torch.tensor([item['target_col'] for item in batch], dtype=torch.long)

    return {
        'row_ids': row_ids,
        'col_ids': col_ids,
        'values': values,
        'target_idx': target_idx,
        'target_value': target_value,
        'target_row': target_row,
        'target_col': target_col
    }
