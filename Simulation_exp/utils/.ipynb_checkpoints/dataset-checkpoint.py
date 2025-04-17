    # utils/dataset.py

import torch
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, x_data, y_data, padding_masks):
        """
        Custom Dataset.
        
        Args:
            x_data (torch.Tensor): Feature data.
            y_data (torch.Tensor or list): Label data.
            padding_masks (torch.Tensor or list): Padding masks, shape [batch_size, seq_length].
        """
        self.x_data = x_data
        self.y_data = y_data
        self.padding_masks = padding_masks

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        return self.x_data[idx], self.padding_masks[idx], self.y_data[idx]
