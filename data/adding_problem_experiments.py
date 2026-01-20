"""
Adding Problem Experiments
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Optional
import os
import sys


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from QDCNN import create_qdcnn_model
from data.adding_problem import load_adding_problem_data

class CheckpointManager:
    
    
    def __init__(self, checkpoint_dir: str = 'checkpoints'):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def save(self, epoch: int, model: nn.Module, optimizer: optim.Optimizer,
             train_losses: List[float], test_losses: List[float],
             config: Dict) -> None:
        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f"adding_T{config['T']}_L{config['L']}_epoch{epoch}.pt"
        )
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': train_losses,
            'test_losses': test_losses,
            'config': config
        }, checkpoint_path)
        
        print(f"  → Checkpoint saved: epoch {epoch}")
    
    def load(self, model: nn.Module, optimizer: optim.Optimizer,
             config: Dict) -> Optional[Tuple[int, List[float], List[float]]]:
        
        checkpoint_path = self._find_latest(config['T'], config['L'])
        if checkpoint_path is None:
            return None
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return (
            checkpoint['epoch'],
            checkpoint['train_losses'],
            checkpoint['test_losses']
        )
    
    def _find_latest(self, T: int, L: int) -> Optional[str]:
        pattern = f"adding_T{T}_L{L}_epoch"
        checkpoints = [
            f for f in os.listdir(self.checkpoint_dir)
            if f.startswith(pattern) and f.endswith('.pt')
        ]
        
        if not checkpoints:
            return None
        
        epochs = [int(f.split('epoch')[1].split('.pt')[0]) for f in checkpoints]
        latest_idx = epochs.index(max(epochs))
        return os.path.join(self.checkpoint_dir, checkpoints[latest_idx])


def train_adding_problem(X_train: List[torch.Tensor],
                        X_test: List[torch.Tensor],
                        y_train: np.ndarray,
                        y_test: np.ndarray,
                        n_layers: int = 3,
                        epochs: int = 100,
                        learning_rate: float = 0.001,
                        batch_size: int = 32,
                        T: int = 600,
                        resume: bool = True) -> Tuple:
    
    input_features = X_train[0].shape[1]  
    n_qubits = 8
    config = {'T': T, 'L': n_layers}
    
    
    model = create_qdcnn_model(
        task_type='regression',
        input_features=input_features,
        output_features=1,
        n_qubits=n_qubits,
        n_layers=n_layers
    )
    X_train_tensor = torch.stack(X_train)
    X_test_tensor = torch.stack(X_test)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Checkpoint management
    checkpoint_manager = CheckpointManager()
    start_epoch = 0
    train_losses, test_losses = [], []
    
    if resume:
        checkpoint_data = checkpoint_manager.load(model, optimizer, config)
        if checkpoint_data:
            start_epoch, train_losses, test_losses = checkpoint_data
            start_epoch += 1
            print(f"\n  Resumed from epoch {start_epoch-1}")
            print(f"  Train loss: {train_losses[-1]:.4f}")
            print(f"  Test loss: {test_losses[-1]:.4f}\n")
    
    # Training loop
    print(f"\n{'Epoch':<8} {'Train Loss':<12} {'Test Loss':<12}")
    print("-" * 35)
    
    for epoch in range(start_epoch, epochs):
        # Training
        model.train()
        epoch_train_loss = 0.0
        n_train_batches = 0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
            n_train_batches += 1
        
        avg_train_loss = epoch_train_loss / n_train_batches
        
        # Evaluation
        model.eval()
        epoch_test_loss = 0.0
        n_test_batches = 0
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                epoch_test_loss += loss.item()
                n_test_batches += 1
        
        avg_test_loss = epoch_test_loss / n_test_batches
        
        train_losses.append(avg_train_loss)
        test_losses.append(avg_test_loss)
        
        # Print and save
        if epoch % 5 == 0:
            print(f'{epoch:<8} {avg_train_loss:<12.4f} {avg_test_loss:<12.4f}')
            checkpoint_manager.save(epoch, model, optimizer, 
                                   train_losses, test_losses, config)
    
    # Final checkpoint
    if (epochs - 1) % 5 != 0:
        checkpoint_manager.save(epochs - 1, model, optimizer,
                               train_losses, test_losses, config)
    
    print(f'{epochs-1:<8} {train_losses[-1]:<12.4f} {test_losses[-1]:<12.4f}')
    
    return model, train_losses, test_losses


def run_experiment(T: int, L: int, resume: bool = True) -> Dict:
    """Run single experiment configuration."""
    print(f"\n{'='*60}")
    print(f"  EXPERIMENT: T={T}, Layers={L}")
    print(f"{'='*60}")
    
    # Load data
    X_train, X_test, y_train, y_test = load_adding_problem_data(
        n_train=5000, n_test=1000, seq_length=T
    )
    

    model, train_losses, test_losses = train_adding_problem(
        X_train, X_test, y_train, y_test,
        n_layers=L, epochs=100, learning_rate=0.001,
        batch_size=32, T=T, resume=resume
    )
    
    # Results
    print(f"\n{'='*60}")
    print(f"  FINAL RESULTS (T={T}, L={L})")
    print(f"  Train Loss: {train_losses[-1]:.4f}")
    print(f"  Test Loss:  {test_losses[-1]:.4f}")
    
    return {
        'model': model,
        'train_losses': train_losses,
        'test_losses': test_losses,
        'T': T,
        'L': L
    }


def plot_results(results: Dict[str, Dict], save_path: str = 'results/adding_problem.png'):
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('QDCNN on Adding Problem: Systematic Analysis',
                 fontsize=16, fontweight='bold')
    
    configs = [
        ('T=200, L=3', 200, 3),
        ('T=200, L=4', 200, 4),
        ('T=600, L=3', 600, 3),
        ('T=600, L=4', 600, 4)
    ]
    
    for idx, (label, T, L) in enumerate(configs):
        ax = axes[idx // 2, idx % 2]
        key = f'T{T}_L{L}'
        
        if key in results:
            data = results[key]
            epochs = range(len(data['train_losses']))
            
            ax.plot(epochs, data['train_losses'], label='Train Loss',
                   linewidth=2.5, alpha=0.8, color='#2E86AB')
            ax.plot(epochs, data['test_losses'], label='Test Loss',
                   linewidth=2.5, alpha=0.8, color='#F18F01')
            
            ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
            ax.set_ylabel('Loss (MSE)', fontsize=11, fontweight='bold')
            ax.set_title(
                f'{label}\nFinal - Train: {data["train_losses"][-1]:.4f}, '
                f'Test: {data["test_losses"][-1]:.4f}',
                fontsize=11
            )
            ax.legend(loc='upper right', fontsize=10)
            ax.grid(True, alpha=0.3)
    
    
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nPlot saved to {save_path}")


def main():
    """Run all Adding Problem experiments."""
    print("="*60)
    print("  QUANTUM DILATED CNN - ADDING PROBLEM")
    print("="*60)
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    # Set seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Experiment matrix
    experiments = [
        (200, 3),
        (200, 4),
        (600, 3),
        (600, 4)
    ]
    
    results = {}
    
    for T, L in experiments:
        try:
            results[f'T{T}_L{L}'] = run_experiment(T, L, resume=True)
        except Exception as e:
            print(f"\nError in T={T}, L={L}: {str(e)}")
            continue
    if results:
        print("\n" + "="*60)
        print("  SUMMARY TABLE")
        print("="*60)
        print(f"{'Config':<12} {'Train Loss':<12} {'Test Loss':<12}")
        print("-"*40)
        
        for key, data in results.items():
            T, L = data['T'], data['L']
            print(f"T={T}, L={L:<3}  "
                  f"{data['train_losses'][-1]:<12.4f} "
                  f"{data['test_losses'][-1]:<12.4f}")
        plot_results(results)
    
    return results


if __name__ == "__main__":
    results = main()