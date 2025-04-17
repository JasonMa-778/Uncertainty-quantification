import torch
from torch.utils.data import Dataset
import numpy as np
import json
import random
class EHRDataset(Dataset):
    def __init__(self, path_documents="mimic3/events.json", path_labels="mimic3/targets.json", path_tokenizer="mimic3/tokenizer.json", mode="train", sequence_length=100, truncation=None, random_seed=None):
        assert mode in ["train", "test"]
        self.sequence_length = sequence_length
        self.truncation = truncation if truncation else []
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)
        with open(path_documents) as f:
            self.data = json.load(f)
        with open(path_labels) as f:
            self.targets = json.load(f)
        with open(path_tokenizer) as f:
            self.tokenizer = json.load(f)
        ref_k = list(self.data.keys()).copy()
        if mode == "train":
            for k in ref_k:
                if k.endswith("8") or k.endswith("9"):
                    del self.data[k]
                    del self.targets[k]
        else:
            for k in ref_k:
                if not k.endswith("8") and not k.endswith("9"):
                    del self.data[k]
                    del self.targets[k]
        self.icu_stays_id = list(self.data.keys())
        assert len(self.data) == len(self.targets)
        if self.truncation:
            self._balance_dataset_by_truncation()
    def _balance_dataset_by_truncation(self):
        bins = [0] + self.truncation
        bin_indices = {i: [] for i in range(len(bins) - 1)}
        seq_lengths = []
        for patient_id in self.icu_stays_id:
            patient_dict = self.data[patient_id]
            seq_length = sum(len(events) for events in patient_dict.values())
            seq_lengths.append((patient_id, seq_length))
        for patient_id, seq_length in seq_lengths:
            for i in range(len(bins) - 1):
                if bins[i] <= seq_length < bins[i + 1]:
                    bin_indices[i].append(patient_id)
                    break
        min_samples = min(len(ids) for ids in bin_indices.values() if ids)
        balanced_ids = []
        for i, ids in bin_indices.items():
            if len(ids) > 0:
                balanced_ids.extend(random.sample(ids, min_samples))
        self.icu_stays_id = balanced_ids
    def __len__(self):
        return len(self.icu_stays_id)
    def __getitem__(self, index):
        patient_dict = self.data[self.icu_stays_id[index]]
        valid_times = []
        valid_events = []
        for t_str, ev_list in patient_dict.items():
            try:
                t_float = float(t_str)
                if np.isnan(t_float):
                    continue
            except:
                continue
            if ev_list is None or len(ev_list) == 0:
                continue
            valid_times.append(t_float)
            valid_events.append(ev_list)
        valid_times.append(179.)
        valid_events.append([[1, 0]])
        if len(valid_times) == 0:
            valid_times = [0.]
            valid_events = [[[1, 0]]]
        minutes = np.repeat(valid_times, list(map(len, valid_events)))
        minutes = torch.tensor(minutes).long()
        codes_list = []
        values_list = []
        for ev_list in valid_events:
            for e in ev_list:
                if e is None:
                    codes_list.append(self.tokenizer.get(str(1), len(self.tokenizer)))
                    values_list.append(0.)
                else:
                    c = e[0] if e[0] is not None else 1
                    v = e[1] if e[1] is not None else 0.0
                    codes_list.append(self.tokenizer.get(str(c), len(self.tokenizer)))
                    values_list.append(v)
        codes = torch.tensor(codes_list).long()
        values = torch.tensor(values_list, dtype=torch.float32)
        seq_l = minutes.size(0)
        if seq_l > self.sequence_length:
            minutes = minutes[-self.sequence_length:]
            codes = codes[-self.sequence_length:]
            values = values[-self.sequence_length:]
            seq_l = self.sequence_length
        padding_length = self.sequence_length - seq_l
        if padding_length > 0:
            minutes = torch.nn.functional.pad(minutes, (padding_length, 0))
            codes = torch.nn.functional.pad(codes, (padding_length, 0))
            values = torch.nn.functional.pad(values, (padding_length, 0))
        attention_mask = torch.zeros(self.sequence_length, dtype=torch.long)
        attention_mask[-seq_l:] = 1
        sample = {
            "codes": codes,
            "values": values,
            "minutes": minutes,
            "attention_mask": attention_mask,
            "target": 1 - self.targets[self.icu_stays_id[index]],
            "seq_l": seq_l
        }
        return sample