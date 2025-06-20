import pennylane as qml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

def quantum_dilated_CNN_circuit(inputs, weights, n_qubits):
    
    
    n_layers = 5  
    weights_per_layer = n_qubits * 2
    for i in range(n_qubits):
        qml.Hadamard(wires=i)
        
    # temporal encoding 
    for i in range(n_qubits):
        temporal_phase = inputs[i] * np.pi * (2*i + 1) / n_qubits
        qml.PhaseShift(temporal_phase, wires=i)
    
    
    for i in range(n_qubits):
        qml.RY(inputs[i] * np.pi, wires=i)
    
    for layer in range(n_layers):
        layer_start = layer * weights_per_layer
        rotation_weights = weights[layer_start:layer_start + n_qubits]
        phase_weights = weights[layer_start + n_qubits:layer_start + 2*n_qubits]
        
        # dilation to capture very long-term dependencies
        dilation = 2 ** (2 * layer)
        for i in range(max(1, n_qubits - dilation)):
            qml.CNOT(wires=[i, min(i + dilation, n_qubits-1)])
        for i in range(n_qubits):
            position_factor = np.sin((i + 1) / n_qubits * np.pi)
            qml.RY(rotation_weights[i] * position_factor, wires=i)
            qml.PhaseShift(phase_weights[i] * position_factor, wires=i)
    
    
    for i in range(n_qubits-1):
        qml.CNOT(wires=[i, i+1])
    return [
        qml.expval(qml.PauliZ(0)), 
        qml.expval(qml.PauliX(1)), 
        qml.expval(qml.PauliY(2)) if n_qubits > 2 else qml.expval(qml.PauliZ(1))
    ]

class HybirdQuantumDCNN(nn.Module):
    def __init__(self, n_qubits, input_features, n_layers=5):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.input_features = input_features
        self.feature_reducer = nn.Sequential(
            nn.Linear(input_features, n_qubits),
            nn.ReLU()
        )
        
        
        weight_shapes = {"weights": (n_qubits * 2 * n_layers,)}
        dev = qml.device("default.qubit", wires=n_qubits)
        qnode = qml.QNode(
            lambda inputs, weights: quantum_dilated_CNN_circuit(inputs, weights, n_qubits), 
            dev
        )
        
        # quantum layer
        self.qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)
        self.post_processing = nn.Sequential(
            nn.Linear(3, 32),  
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1)
        )
        
    def forward(self, x):
        batch_size, timesteps, features = x.shape
        q_out = torch.zeros(batch_size, 3, device=x.device)
        for i in range(batch_size):
            sample_aggregated = x[i].mean(dim=0)
            sample_reduced = self.feature_reducer(sample_aggregated)
            q_out[i] = self.qlayer(sample_reduced)
        
        
        return self.post_processing(q_out)

def train_quantum_model(X_train, X_valid, y_train, y_valid, epochs=200, learning_rate=0.001, task_name="Unknown"):
    
    print(f"\n=== Training Quantum Model for {task_name} ===")
    
    
    input_features = X_train[0].shape[1]
    n_qubits = min(8, input_features)  # Limit qubits to 8
    
    print(f"Model Configuration:")
    print(f"- Input features: {input_features}")
    print(f"- Number of qubits: {n_qubits}")
    print(f"- Training samples: {len(X_train)}")
    print(f"- Validation samples: {len(X_valid)}")
    
    
    model = HybirdQuantumDCNN(n_qubits, input_features)
    X_train_tensor = torch.stack(X_train)
    X_valid_tensor = torch.stack(X_valid)
    if y_train is not None:
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
        y_valid_tensor = torch.tensor(y_valid, dtype=torch.float32).unsqueeze(1)
    else:
        y_train_tensor = torch.randn(len(X_train), 1)
        y_valid_tensor = torch.randn(len(X_valid), 1)
    
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train_losses = []
    val_losses = []
    
    print("\nTraining Progress:")
    for epoch in range(epochs):
        
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            val_outputs = model(X_valid_tensor)
            val_loss = criterion(val_outputs, y_valid_tensor)
        
        train_losses.append(loss.item())
        val_losses.append(val_loss.item())
        if epoch % 20 == 0:
            print(f'Epoch [{epoch:3d}/{epochs}] | '
                  f'Train Loss: {loss.item():.4f} | '
                  f'Val Loss: {val_loss.item():.4f}')
    
    print(f"\nTraining completed for {task_name}!")
    print(f"Final Train Loss: {train_losses[-1]:.4f}")
    print(f"Final Val Loss: {val_losses[-1]:.4f}")
    
    return model, train_losses, val_losses

def evaluate_model(model, X_test, y_test=None, task_name="Unknown"):
    
    print(f"\n=== Evaluating Model on {task_name} ===")
    
    X_test_tensor = torch.stack(X_test)
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        if y_test is not None:
            y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
            test_loss = nn.MSELoss()(test_outputs, y_test_tensor)
            mae = torch.mean(torch.abs(test_outputs - y_test_tensor)).item()
            print(f"Test Results:")
            print(f"- Test Loss (MSE): {test_loss.item():.4f}")
            print(f"- Test MAE: {mae:.4f}")
            return test_loss.item()
        else:
            print(f"Test completed. Output shape: {test_outputs.shape}")
            return test_outputs

def plot_training_curves(train_losses, val_losses, task_name="Unknown"):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss', alpha=0.8)
    plt.plot(val_losses, label='Validation Loss', alpha=0.8)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training Progress - {task_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def get_model_info():
    
    info = {
        'model_name': 'Quantum Dilated convolutional neural network',
        'quantum_device': 'default.qubit',
        'n_qubits': 8,
        'n_layers': 5,
        'circuit_features': [
            'Enhanced temporal encoding',
            'Exponential dilation patterns', 
            'Position-aware rotations',
            'Multi-observable measurements'
        ]
    }
    return info