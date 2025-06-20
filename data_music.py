import numpy as np
import torch
import scipy.io as sio
import os

def load_nottingham_data(filepath='Nottingham.mat'):
   
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found. Generating synthetic music data...")
        return generate_synthetic_music_data()
    
    try:
        # load the data
        data = sio.loadmat(filepath)
        X_train = data['traindata'][0]
        X_valid = data['validdata'][0]
        X_test = data['testdata'][0]
        
        
        print("Raw Nottingham data shapes:")
        print(f"Train data shapes: {[x.shape for x in X_train[:3]]}")
        print(f"Valid data shapes: {[x.shape for x in X_valid[:3]]}")
        print(f"Test data shapes: {[x.shape for x in X_test[:3]]}")
        
        def process_dataset(dataset, max_length=None):
            processed = []
            if max_length is None:
                # the maximum length 
                max_length = max(x.shape[0] for x in dataset)
            
            for sample in dataset:
                if sample.shape[0] < max_length:
                    pad_amount = max_length - sample.shape[0]
                    padded_sample = np.pad(sample, ((0, pad_amount), (0, 0)), mode='constant')
                else:
                    padded_sample = sample[:max_length, :]
                
                
                tensor_sample = torch.tensor(padded_sample.astype(np.float32))
                for col in range(tensor_sample.shape[1]):
                    col_data = tensor_sample[:, col]
                    if col_data.std() > 1e-7:
                        tensor_sample[:, col] = (col_data - col_data.mean()) / col_data.std()
                
                processed.append(tensor_sample)
            return processed
        

        max_train_length = max(x.shape[0] for x in X_train)
        max_valid_length = max(x.shape[0] for x in X_valid)
        max_test_length = max(x.shape[0] for x in X_test)
        max_overall_length = max(max_train_length, max_valid_length, max_test_length)
        X_train = process_dataset(X_train, max_overall_length)
        X_valid = process_dataset(X_valid, max_overall_length)
        X_test = process_dataset(X_test, max_overall_length)
        return X_train, X_valid, X_test
        
    except Exception as e:
        print(f"Error loading Nottingham data: {e}")
        print("Generating synthetic music data instead...")
        return generate_synthetic_music_data()

def generate_synthetic_music_data(n_train=700, n_valid=150, n_test=150, 
                                 sequence_length=200, n_features=8):
    
    print("Generating synthetic music data...")
    
    def generate_music_sequence(length, features):
        time_steps = np.linspace(0, 4*np.pi, length)
        sequence = np.zeros((length, features))
        for f in range(features):
            
            freq1 = 0.5 + f * 0.2
            freq2 = 1.0 + f * 0.3
            component1 = np.sin(freq1 * time_steps + f * np.pi/4)
            component2 = 0.5 * np.sin(freq2 * time_steps + f * np.pi/3)
            noise = 0.2 * np.random.randn(length)
            trend = 0.1 * np.linspace(-1, 1, length)
            
            sequence[:, f] = component1 + component2 + noise + trend
        
        # Normalize
        sequence = (sequence - sequence.mean(axis=0)) / (sequence.std(axis=0) + 1e-7)
        
        return torch.tensor(sequence.astype(np.float32))
    
    
    X_train = [generate_music_sequence(sequence_length, n_features) for _ in range(n_train)]
    X_valid = [generate_music_sequence(sequence_length, n_features) for _ in range(n_valid)]
    X_test = [generate_music_sequence(sequence_length, n_features) for _ in range(n_test)]
    
    print("Synthetic Music Dataset Characteristics:")
    print(f"Training samples: {len(X_train)}")
    print(f"Training sample shape: {X_train[0].shape}")
    print(f"Validation samples: {len(X_valid)}")
    print(f"Test samples: {len(X_test)}")
    
    return X_train, X_valid, X_test

def get_music_info():
    
    info = {
        'task_name': 'Music Modeling',
        'description': 'Model temporal patterns in musical sequences',
        'data_source': 'Nottingham Folk Music Dataset (or synthetic)',
        'input_features': 'Variable (typically 8 for synthetic)',
        'output_dimension': 'Sequence modeling',
        'difficulty': 'Complex temporal and harmonic patterns'
    }
    return info