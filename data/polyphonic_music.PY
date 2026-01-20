"""
polyphonic music sequences
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from scipy.io import loadmat
from typing import Tuple, List, Optional
import os


class PolyphonicMusicDataset(Dataset):
   
    
    def __init__(self, sequences: List, seq_length: int = 32):
        
        self.sequences = []
        self.seq_length = seq_length
        
        for seq in sequences:
            if len(seq) > seq_length:
                for i in range(len(seq) - seq_length):
                    input_seq = seq[i:i+seq_length]
                    target = seq[i+seq_length]
                    self.sequences.append((input_seq, target))
    
    def __len__(self) -> int:
        return len(self.sequences)
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        input_seq, target = self.sequences[idx]
        return input_seq.float(), target.float()


def load_polyphonic_music_data(dataset: str, 
                               data_dir: str = './mdata') -> Tuple[List, List, List]:
   
    dataset_files = {
        "JSB": "JSB_Chorales.mat",
        "Muse": "MuseData.mat",
        "Nott": "Nottingham.mat",
        "Piano": "Piano_midi.mat"
    }
    
    if dataset not in dataset_files:
        raise ValueError(
            f"Unknown dataset: {dataset}. "
            f"Choose from: {list(dataset_files.keys())}"
        )
    
    filepath = os.path.join(data_dir, dataset_files[dataset])
    print(f'Loading {dataset} from {filepath}...')
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Data file not found: {filepath}\n"
            f"Please download the dataset and place it in {data_dir}/"
        )
    
    data = loadmat(filepath)
    X_train = data['traindata'][0]
    X_valid = data['validdata'][0]
    X_test = data['testdata'][0]
    for data_split in [X_train, X_valid, X_test]:
        for i in range(len(data_split)):
            data_split[i] = torch.Tensor(data_split[i].astype(np.float64))
    
    return X_train, X_valid, X_test


class PolyphonicMusicConfig:
    DATASETS = {
        'JSB': {
            'name': 'JSB Chorales',
            'n_features': 88,  # piano keys
            'description': 'Bach chorales'
        },
        'Muse': {
            'name': 'MuseData',
            'n_features': 88,
            'description': 'Classical music'
        },
        'Nott': {
            'name': 'Nottingham',
            'n_features': 88,
            'description': 'Folk tunes'
        },
        'Piano': {
            'name': 'Piano-midi',
            'n_features': 88,
            'description': 'Piano performances'
        }
    }
    
    @staticmethod
    def get_dataset_info(dataset: str) -> dict:
        """Get information about a dataset."""
        return PolyphonicMusicConfig.DATASETS.get(
            dataset, 
            {'name': 'Unknown', 'n_features': 88, 'description': ''}
        )


def compute_nll(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    
    probs = torch.clamp(outputs, min=1e-7, max=1-1e-7)
    nll = -(targets * torch.log(probs) + (1 - targets) * torch.log(1 - probs))
    
    return nll.mean()


def get_data_statistics(sequences: List[torch.Tensor]) -> dict:
    
    lengths = [len(seq) for seq in sequences]
    features = sequences[0].shape[1] if len(sequences) > 0 else 0
    total_elements = sum(seq.numel() for seq in sequences)
    active_elements = sum((seq > 0).sum().item() for seq in sequences)
    sparsity = active_elements / total_elements if total_elements > 0 else 0
    
    return {
        'n_sequences': len(sequences),
        'n_features': features,
        'min_length': min(lengths) if lengths else 0,
        'max_length': max(lengths) if lengths else 0,
        'mean_length': np.mean(lengths) if lengths else 0,
        'sparsity': sparsity
    }


if __name__ == "__main__":
    
    print(" Polyphonic Music Dataset")
    print("=" * 60)
    try:
        X_train, X_valid, X_test = load_polyphonic_music_data('Nott')
        
        print("\nDataset loaded successfully!")
        print(f"Train sequences: {len(X_train)}")
        print(f"Valid sequences: {len(X_valid)}")
        print(f"Test sequences: {len(X_test)}")
        train_stats = get_data_statistics(X_train)
        print(f"\nTraining set statistics:")
        for key, value in train_stats.items():
            print(f"  {key}: {value}")
        dataset = PolyphonicMusicDataset(X_train, seq_length=32)
        print(f"\nDataset samples: {len(dataset)}")
        
        sample_input, sample_target = dataset[0]
        print(f"Input shape: {sample_input.shape}")
        print(f"Target shape: {sample_target.shape}")
        
    