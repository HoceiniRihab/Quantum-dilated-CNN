import numpy as np
import torch

def generate_adding_problem(n_samples=1000, sequence_length=600):
    
    X = np.random.uniform(0, 1, (n_samples, sequence_length))
    indicators = np.zeros((n_samples, sequence_length))
    for i in range(n_samples):
        idx1, idx2 = np.random.choice(sequence_length, size=2, replace=False)
        indicators[i, idx1] = 1
        indicators[i, idx2] = 1
    

    y = np.sum(X * indicators, axis=1)
    X_combined = np.column_stack((X, indicators))
    
    return X_combined, y

def load_adding_problem_data(n_samples=1000, seq_length=600, train_ratio=0.7, valid_ratio=0.15):
   
    print(f"Generating adding problem dataset with t={seq_length}...")
    X, y = generate_adding_problem(n_samples, seq_length)
    
    # Split into train/valid/test
    train_size = int(train_ratio * n_samples)
    valid_size = int(valid_ratio * n_samples)
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_valid = X[train_size:train_size+valid_size]
    y_valid = y[train_size:train_size+valid_size]
    X_test = X[train_size+valid_size:]
    y_test = y[train_size+valid_size:]
    
    def to_tensor_list(data):
        return [torch.tensor(sample.reshape(-1, 2).astype(np.float32)) for sample in data]
    
    X_train = to_tensor_list(X_train)
    X_valid = to_tensor_list(X_valid)
    X_test = to_tensor_list(X_test)
    
    
    print("Adding Problem Dataset Characteristics:")
    print(f"Training samples: {len(X_train)}")
    print(f"Training sample shape: {X_train[0].shape}")
    print(f"Validation samples: {len(X_valid)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Target range: [{y_train.min():.3f}, {y_train.max():.3f}]")
    
    return X_train, X_valid, X_test, y_train, y_valid, y_test

def get_adding_problem_info():
    
    info = {
        'task_name': 'Adding Problem',
        'description': 'Identify and sum two marked numbers in a long sequence',
        'sequence_length': 600,
        'input_features': 2,  # sequence value + indicator
        'output_dimension': 1,
        'difficulty': 'Long-term temporal dependency'
    }
    return info