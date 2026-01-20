"""
Polyphonic Music Modeling Experiments

"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional, List
import os
import pickle
import time
import sys


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from QDCNN import create_qdcnn_model
from data.polyphonic_music import (
    load_polyphonic_music_data,
    PolyphonicMusicDataset,
    compute_nll,
    get_data_statistics
)


class MusicCheckpointManager:
   
    def __init__(self, checkpoint_dir: str = './checkpoints'):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def get_path(self, dataset_name: str, n_qubits: int, n_layers: int) -> str:
        
        return os.path.join(
            self.checkpoint_dir,
            f"{dataset_name}_q{n_qubits}_l{n_layers}_checkpoint.pkl"
        )
    
    def save(self, checkpoint_path: str, state: Dict) -> None:
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(state, f)
        print(f" Checkpoint saved: {checkpoint_path}")
    
    def load(self, checkpoint_path: str) -> Optional[Dict]:
        if os.path.exists(checkpoint_path):
            with open(checkpoint_path, 'rb') as f:
                state = pickle.load(f)
            print(f" Checkpoint loaded: {checkpoint_path}")
            return state
        return None
    
    def remove(self, checkpoint_path: str) -> None:
        
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
            print(f"✓ Checkpoint removed: {checkpoint_path}")


def train_epoch(model: nn.Module,
                train_loader: DataLoader,
                optimizer: optim.Optimizer,
                device: torch.device,
                epoch: int,
                print_every: int = 50) -> float:
    
    model.train()
    running_nll = 0.0
    batch_nlls = []
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        nll = compute_nll(outputs, targets)
        nll.backward()
        optimizer.step()
        
        running_nll += nll.item()
        batch_nlls.append(nll.item())
        
        if (batch_idx + 1) % print_every == 0:
            avg_nll = running_nll / print_every
            print(f"  Epoch {epoch} | Batch {batch_idx + 1}/{len(train_loader)} "
                  f"| NLL: {avg_nll:.4f}")
            running_nll = 0.0
    
    return np.mean(batch_nlls)


def evaluate(model: nn.Module,
            data_loader: DataLoader,
            device: torch.device) -> float:
    
    model.eval()
    total_nll = 0.0
    num_batches = 0
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            nll = compute_nll(outputs, targets)
            total_nll += nll.item()
            num_batches += 1
    
    return total_nll / num_batches


def run_music_experiment(dataset_name: str = "Nott",
                        n_qubits: int = 8,
                        n_layers: int = 3,
                        batch_size: int = 32,
                        epochs: int = 100,
                        learning_rate: float = 0.001,
                        seq_length: int = 32,
                        print_every: int = 50,
                        checkpoint_dir: str = "./checkpoints",
                        resume: bool = True) -> Optional[Dict]:
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Checkpoint management
    checkpoint_manager = MusicCheckpointManager(checkpoint_dir)
    checkpoint_path = checkpoint_manager.get_path(dataset_name, n_qubits, n_layers)
    
    checkpoint = None
    if resume:
        checkpoint = checkpoint_manager.load(checkpoint_path)
    if checkpoint is None:
        print(f"\n{'='*70}")
        print(f"Quantum Dilated CNN - Polyphonic Music Modeling")
        print(f"{'='*70}")
        print(f"Dataset: {dataset_name}")
        print(f"Device: {device}")
        print(f"{'='*70}\n")
        X_train, X_valid, X_test = load_polyphonic_music_data(dataset_name)
        n_features = X_train[0].shape[1]
        train_dataset = PolyphonicMusicDataset(X_train, seq_length=seq_length)
        val_dataset = PolyphonicMusicDataset(X_valid, seq_length=seq_length)
        test_dataset = PolyphonicMusicDataset(X_test, seq_length=seq_length)
        
        print(f"Dataset Statistics:")
        print(f"  Training sequences: {len(train_dataset)}")
        print(f"  Validation sequences: {len(val_dataset)}")
        print(f"  Test sequences: {len(test_dataset)}")
        print(f"  Sequence length: {seq_length}")
        print(f"  Features : {n_features}")
        print(f"{'='*70}\n")
        model = create_qdcnn_model(
            task_type='binary_classification',
            input_features=n_features,
            output_features=n_features,
            n_qubits=n_qubits,
            n_layers=n_layers,
            hidden_dim=64,
            dropout_rate=0.2
        )
        model = model.to(device)
        print(f"Model Architecture:")
        print(f"  - Qubits: {n_qubits}")
        print(f"  - Quantum Layers: {n_layers}")
        print(f"  - Total Parameters: {model[0].count_parameters():,}")
        print(f"{'='*70}\n")
        
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        start_epoch = 1
        train_nlls = []
        val_nlls = []
        best_val_nll = float('inf')
        best_model_state = None
        start_time = time.time()
        
    else:
        print(f"\n{'='*70}")
        print(f"Resuming Training from Checkpoint")
        print(f"{'='*70}\n")
        X_train, X_valid, X_test = load_polyphonic_music_data(dataset_name)
        n_features = X_train[0].shape[1]
        train_dataset = PolyphonicMusicDataset(X_train, seq_length=seq_length)
        val_dataset = PolyphonicMusicDataset(X_valid, seq_length=seq_length)
        test_dataset = PolyphonicMusicDataset(X_test, seq_length=seq_length)
        model = create_qdcnn_model(
            task_type='binary_classification',
            input_features=n_features,
            output_features=n_features,
            n_qubits=n_qubits,
            n_layers=n_layers,
            hidden_dim=64,
            dropout_rate=0.2
        )
        model = model.to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        train_nlls = checkpoint['train_nlls']
        val_nlls = checkpoint['val_nlls']
        best_val_nll = checkpoint['best_val_nll']
        best_model_state = checkpoint['best_model_state']
        start_time = checkpoint['start_time']
        print(f"Resuming from epoch {start_epoch}/{epochs}")
        print(f"Best validation NLL: {best_val_nll:.4f}\n")
    
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
   
    print("Starting Training\n")
    
    try:
        for epoch in range(start_epoch, epochs + 1):
            epoch_start = time.time()
            train_nll = train_epoch(model, train_loader, optimizer, 
                                   device, epoch, print_every)
            train_nlls.append(train_nll)
            val_nll = evaluate(model, val_loader, device)
            val_nlls.append(val_nll)
            epoch_time = time.time() - epoch_start
            print(f"\n{'─'*70}")
            print(f"Epoch {epoch}/{epochs} Summary (Time: {epoch_time:.2f}s)")
            print(f"{'─'*70}")
            print(f"Train NLL: {train_nll:.4f} | Val NLL: {val_nll:.4f}")
            print(f"{'─'*70}\n")
            if val_nll < best_val_nll:
                best_val_nll = val_nll
                best_model_state = model.state_dict().copy()
                print(f"New best model! (Val NLL: {val_nll:.4f})\n")
            checkpoint_state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_nlls': train_nlls,
                'val_nlls': val_nlls,
                'best_val_nll': best_val_nll,
                'best_model_state': best_model_state,
                'start_time': start_time,
                'n_qubits': n_qubits,
                'n_layers': n_layers,
                'dataset_name': dataset_name
            }
            checkpoint_manager.save(checkpoint_path, checkpoint_state)
    
    except KeyboardInterrupt:
        print("\n\nTraining interrupted. Checkpoint saved.")
        return None
    
    except Exception as e:
        print(f"\n\nError: {e}")
        print("Checkpoint saved for resume.")
        raise
    
    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"Training Complete! Total Time: {total_time/60:.2f} minutes")
    print(f"{'='*70}\n")
    model.load_state_dict(best_model_state)
    print("Computing final NLL on all datasets...\n")
    final_train_nll = evaluate(model, train_loader, device)
    final_val_nll = evaluate(model, val_loader, device)
    final_test_nll = evaluate(model, test_loader, device)
    print(f"\n{'='*70}")
    print(f"FINAL RESULTS")
    print(f"{'='*70}")
    print(f"Train NLL: {final_train_nll:.4f}")
    print(f"Valid NLL: {final_val_nll:.4f}")
    print(f"Test NLL:  {final_test_nll:.4f}")
    print(f"{'='*70}\n")
    checkpoint_manager.remove(checkpoint_path)
    results = {
        'model': model,
        'train_nlls': train_nlls,
        'val_nlls': val_nlls,
        'final_train_nll': final_train_nll,
        'final_val_nll': final_val_nll,
        'final_test_nll': final_test_nll,
        'total_params': model[0].count_parameters(),
        'n_layers': n_layers,
        'dataset': dataset_name
    }
    
    return results
def run_multi_layer_experiments(dataset_name: str = "Piano",
                                n_qubits: int = 8,
                                layer_configs: List[int] = [3, 4],
                                **kwargs) -> Dict:
    
    all_results = {}
    for n_layers in layer_configs:
        print(f"\n{'#'*70}")
        print(f"# Starting Experiment: {n_layers} Quantum Layers")
        print(f"{'#'*70}\n")
        results = run_music_experiment(
            dataset_name=dataset_name,
            n_qubits=n_qubits,
            n_layers=n_layers,
            **kwargs
        )
        
        if results is not None:
            all_results[f'{n_layers}_layers'] = results
            print(f"\n✓ Experiment with {n_layers} layers completed!")
            print(f"  Train NLL: {results['final_train_nll']:.4f}")
            print(f"  Valid NLL: {results['final_val_nll']:.4f}")
            print(f"  Test NLL:  {results['final_test_nll']:.4f}")
            print(f"  Parameters: {results['total_params']:,}\n")
    if len(all_results) > 1:
        print(f"\n{'='*70}")
        print(f"COMPARATIVE RESULTS")
        print(f"{'='*70}")
        for config_name, result in all_results.items():
            print(f"{config_name}:")
            print(f"  Train NLL: {result['final_train_nll']:.4f}")
            print(f"  Valid NLL: {result['final_val_nll']:.4f}")
            print(f"  Test NLL:  {result['final_test_nll']:.4f}")
            print(f"  Params: {result['total_params']:,}")
            print()
        print(f"{'='*70}\n")
    
    return all_results


def main():
    
    print("="*70)
    print("  QUANTUM DILATED CNN - POLYPHONIC MUSIC")
    print("="*70)
    
    all_results = run_multi_layer_experiments(
        dataset_name="Nott",
        n_qubits=8,
        layer_configs=[3, 4],
        batch_size=32,
        epochs=100,
        learning_rate=0.001,
        seq_length=32,
        print_every=50,
        checkpoint_dir="./checkpoints",
        resume=True
    )
    
    print("\n All experiments completed successfully!")
    return all_results


if __name__ == "__main__":
    results = main()