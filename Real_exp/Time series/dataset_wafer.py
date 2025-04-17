import torch
from torch.utils.data import Dataset
import numpy as np
import os

class WaferDataset(Dataset):
    def __init__(self, root_dir, mode="train", sequence_length=152):
        assert mode in ["train", "test"]
        self.sequence_length = sequence_length
        file_path = os.path.join(root_dir, "Wafer_TEST.ts" if mode == "test" else "Wafer_TRAIN.ts")
        self.data, self.targets = self.load_ts_file(file_path)
    def load_ts_file(self, file_path):
        with open(file_path, "r") as f:
            lines = f.readlines()
        data_start = next(i for i, line in enumerate(lines) if "@data" in line) + 1
        raw_data = lines[data_start:]
        sequences, labels = [], []
        for line in raw_data:
            values = line.strip().split(",")
            label_part = values[-1].split(":")
            label = int(label_part[-1])
            labels.append(1 if label == 1 else 0)
            sequence = [float(v) for v in values[:-1]]
            sequences.append(sequence)
        sequences = np.array(sequences, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)
        return sequences, labels
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        sequence = self.data[index]
        target = self.targets[index]
        drop_length = np.random.randint(0, self.sequence_length)
        truncated_sequence = sequence[:-drop_length] if drop_length > 0 else sequence
        actual_length = len(truncated_sequence)
        padded_sequence = np.zeros(self.sequence_length, dtype=np.float32)
        padded_sequence[:actual_length] = truncated_sequence
        attention_mask = np.zeros(self.sequence_length, dtype=np.long)
        attention_mask[:actual_length] = 1
        return {
            "sequence": torch.tensor(padded_sequence, dtype=torch.float32),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "target": torch.tensor(target, dtype=torch.long),
            "sql":actual_length
        }