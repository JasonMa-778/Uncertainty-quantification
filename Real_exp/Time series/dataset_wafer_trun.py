import torch
from torch.utils.data import Dataset
import numpy as np
import os
import random

class WaferDataset(Dataset):
    def __init__(self, root_dir, mode="train", sequence_length=152, truncation=None, random_seed=None):
        
        assert mode in ["train", "test"]
        self.sequence_length = sequence_length
        self.random_seed = random_seed
        file_path = os.path.join(root_dir, "Wafer_TEST.ts" if mode == "train" else "Wafer_TRAIN.ts")
        self.data, self.targets = self.load_ts_file(file_path)
        self.augment_fractions = truncation if truncation is not None else []
        if self.augment_fractions:
            self.samples = []
            for idx in range(len(self.data)):
                for frac in self.augment_fractions:
                    self.samples.append((idx, frac))
        else:
            self.samples = list(range(len(self.data)))
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
        return len(self.samples)
    def __getitem__(self, index):
        if self.augment_fractions:
            idx, frac = self.samples[index]
        else:
            idx = self.samples[index]
            frac = None
        base_sequence = self.data[idx]
        target = self.targets[idx]
        if len(base_sequence) != self.sequence_length:
            if len(base_sequence) > self.sequence_length:
                base_sequence = base_sequence[:self.sequence_length]
            else:
                padded = np.zeros(self.sequence_length, dtype=np.float32)
                padded[:len(base_sequence)] = base_sequence
                base_sequence = padded
        if frac is not None:
            kept_length = int(round(frac * self.sequence_length))
            kept_length = max(1, min(self.sequence_length, kept_length))
            if self.random_seed is not None:
                local_rng = random.Random(self.random_seed + index)
                start_idx = local_rng.randint(0, self.sequence_length - kept_length)
            else:
                start_idx = np.random.randint(0, self.sequence_length - kept_length + 1)
            window = base_sequence[start_idx: start_idx + kept_length]
            padded_sequence = np.zeros(self.sequence_length, dtype=np.float32)
            padded_sequence[:kept_length] = window
            effective_length = kept_length
        else:
            padded_sequence = base_sequence.copy()
            effective_length = self.sequence_length
        attention_mask = np.zeros(self.sequence_length, dtype=np.int64)
        attention_mask[:effective_length] = 1
        return {
            "sequence": torch.tensor(padded_sequence, dtype=torch.float32),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "target": torch.tensor(target, dtype=torch.long),
            "sql": effective_length
        }