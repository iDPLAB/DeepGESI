# Copyright (c) the Lab of Intelligent Data Processing, Wakayama University.
# All rights reserved.

#Dataset_Loader.py
import os
import json
import torch
import numpy as np
from scipy.io import wavfile
from torch.utils.data import Dataset, random_split
import torchaudio.transforms as T

class GESIDataset(Dataset):
    def __init__(self, json_path, wav_root, target_sr=16000, transform=None):
        self.wav_root = wav_root
        self.target_sr = target_sr
        self.transform = transform

        with open(json_path, 'r') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        signal_id = item["signal"]
        gesi_score = float(item["gesi_score"])

        wav_path = os.path.join(self.wav_root, signal_id + ".wav")
        sr, waveform = wavfile.read(wav_path)

    
        if waveform.ndim == 2:
            waveform = waveform.mean(axis=1)

        # Convert to float32 and normalize to [-1, 1]
        waveform = waveform.astype(np.float32) / 32768.0
        waveform = torch.tensor(waveform).unsqueeze(0)  # [1, T]

        
        if sr != self.target_sr:
            resampler = T.Resample(orig_freq=sr, new_freq=self.target_sr)
            waveform = resampler(waveform)

        if self.transform:
            waveform = self.transform(waveform)

        return waveform, torch.tensor([gesi_score], dtype=torch.float32)

def split_dataset(json_path, wav_root, target_sr=16000, seed=9999,
                  train_ratio=0.8, val_ratio=0.1):
    dataset = GESIDataset(json_path, wav_root, target_sr)

    total_len = len(dataset)
    train_len = int(total_len * train_ratio)
    val_len = int(total_len * val_ratio)
    test_len = total_len - train_len - val_len

    return random_split(dataset, [train_len, val_len, test_len],
                        generator=torch.Generator().manual_seed(seed))


def collate_fn_pad(batch):
    import torch.nn.functional as F
    waveforms, scores = zip(*batch)
    lengths = [w.shape[-1] for w in waveforms]
    max_len = max(lengths)

    ## If all sample lengths are the same, stack directly without padding

    if all(l == max_len for l in lengths):
        return torch.stack(waveforms), torch.stack(scores)

    # Otherwise, perform padding
    padded_waveforms = [F.pad(w, (0, max_len - w.shape[-1])) for w in waveforms]
    padded_waveforms = torch.stack(padded_waveforms)
    scores = torch.stack(scores)

    return padded_waveforms, scores

