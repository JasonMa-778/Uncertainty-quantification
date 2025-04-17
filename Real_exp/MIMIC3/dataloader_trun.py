import torch
from torch.utils.data import Dataset
import numpy as np
import json
import random
class EHRDataset(Dataset):
    def __init__(self,
                 path_documents="events.json",
                 path_labels="targets.json",
                 path_tokenizer="tokenizer.json",
                 mode="train",
                 sequence_length=100,
                 truncation=None,
                 random_seed=None):
        assert mode in ["train", "test"]
        self.sequence_length = sequence_length
        self.augment_fractions = truncation if truncation is not None else []
        self.random_seed = random_seed
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
                if not (k.endswith("8") or k.endswith("9")):
                    del self.data[k]
                    del self.targets[k]
        valid_ids = []
        for patient_id in list(self.data.keys()):
            patient_dict = self.data[patient_id]
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
            token_count = sum(len(ev_list) for ev_list in valid_events)
            if token_count >= self.sequence_length:
                valid_ids.append(patient_id)
        self.samples = []
        if self.augment_fractions:
            for patient_id in valid_ids:
                for frac in self.augment_fractions:
                    self.samples.append((patient_id, frac))
        else:
            self.samples = valid_ids
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, index):
        if self.augment_fractions:
            patient_id, frac = self.samples[index]
        else:
            patient_id = self.samples[index]
            frac = None
        patient_dict = self.data[patient_id]
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
        repeat_counts = list(map(len, valid_events))
        minutes = np.repeat(valid_times, repeat_counts)
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
        elif seq_l < self.sequence_length:
            padding_length = self.sequence_length - seq_l
            minutes = torch.nn.functional.pad(minutes, (0, padding_length))
            codes = torch.nn.functional.pad(codes, (0, padding_length))
            values = torch.nn.functional.pad(values, (0, padding_length))
        base_seq = {
            "minutes": minutes,
            "codes": codes,
            "values": values
        }
        if frac is not None:
            kept_length = int(round(frac * self.sequence_length))
            kept_length = max(1, min(self.sequence_length, kept_length))
            if self.random_seed is not None:
                local_rng = random.Random(self.random_seed + index)
                start_idx = local_rng.randint(0, self.sequence_length - kept_length)
            else:
                start_idx = random.randint(0, self.sequence_length - kept_length)
            minutes_aug = base_seq["minutes"][start_idx: start_idx + kept_length]
            codes_aug = base_seq["codes"][start_idx: start_idx + kept_length]
            values_aug = base_seq["values"][start_idx: start_idx + kept_length]
            padding_length = self.sequence_length - kept_length
            minutes = torch.nn.functional.pad(minutes_aug, (0, padding_length))
            codes = torch.nn.functional.pad(codes_aug, (0, padding_length))
            values = torch.nn.functional.pad(values_aug, (0, padding_length))
            effective_seq_l = kept_length
        else:
            minutes = base_seq["minutes"]
            codes = base_seq["codes"]
            values = base_seq["values"]
            effective_seq_l = self.sequence_length
        attention_mask = torch.zeros(self.sequence_length, dtype=torch.long)
        attention_mask[:effective_seq_l] = 1
        sample = {
            "codes": codes,
            "values": values,
            "minutes": minutes,
            "attention_mask": attention_mask,
            "target": 1 - self.targets[patient_id],
            "seq_l": effective_seq_l
        }
        return sample