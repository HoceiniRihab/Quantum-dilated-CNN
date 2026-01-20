"""
Adding Problem Dataset
Temporal sequence learning benchmark task
"""

import numpy as np
import torch
from typing import Tuple, List


def generate_adding_problem(n_samples: int, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate the Adding Problem dataset
    Task: Given a sequence of random numbers and binary indicators,
    sum the two numbers where the indicators are 1
    
    """
   
    X = np.random.uniform(0, 1, (n_samples, seq_length))
    indicators = np.zeros((n_samples, seq_length))
    for i in range(n_samples):
        idx1, idx2 = np.random.choice(seq_length, size=2, replace=False)
        indicators[i, idx1] = 1
        indicators[i, idx2] = 1
    y = np.sum(X * indicators, axis=1)
    X_combined = np.stack((X, indicators), axis=-1)
    
    return X_combined, y


def load_adding_problem_data(n_train: int = 5000,
                            n_test: int = 1000,
                            seq_length: int = 600) -> Tuple[List[torch.Tensor], 
                                                            List[torch.Tensor],
                                                            np.ndarray, 
                                                            np.ndarray]:
    
    X_train, y_train = generate_adding_problem(n_train, seq_length)
    X_test, y_test = generate_adding_problem(n_test, seq_length)
    to_tensor = lambda data: [
        torch.tensor(s, dtype=torch.float32) for s in data
    ]
    
    X_train_tensors = to_tensor(X_train)
    X_test_tensors = to_tensor(X_test)
    
    return X_train_tensors, X_test_tensors, y_train, y_test


class AddingProblemConfig:
    
    EASY = {'seq_length': 200, 'description': 'Easy (T=200)'}
    MEDIUM = {'seq_length': 400, 'description': 'Medium (T=400)'}
    HARD = {'seq_length': 600, 'description': 'Hard (T=600)'}
    
    @staticmethod
    def get_config(difficulty: str) -> dict:
        
        configs = {
            'easy': AddingProblemConfig.EASY,
            'medium': AddingProblemConfig.MEDIUM,
            'hard': AddingProblemConfig.HARD
        }
        return configs.get(difficulty.lower(), AddingProblemConfig.HARD)




if __name__ == "__main__":
    
    print("Testing Adding Problem Dataset Generation")
    print("=" * 50)
    
    X_train, X_test, y_train, y_test = load_adding_problem_data(
        n_train=100, n_test=20, seq_length=50
    )
    
    print(f"Train samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Sample shape: {X_train[0].shape}")
    print(f"Target shape: {y_train.shape}")
    print(f"\nSample target values: {y_train[:5]}")
    print(f"Target range: [{y_train.min():.3f}, {y_train.max():.3f}]")
    