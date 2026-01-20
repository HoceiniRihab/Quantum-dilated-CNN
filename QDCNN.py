"""
Quantum Dilated Convolutional Neural Network (QDCNN)
"""

import pennylane as qml
import torch
import torch.nn as nn
import numpy as np
from typing import List


def quantum_dilated_CNN_circuit(inputs: torch.Tensor, 
                                weights: torch.Tensor,
                                n_qubits: int, 
                                n_layers: int) -> List:
 
    weights_per_layer = n_qubits * 2
    
    # encoding
    for i in range(n_qubits):
        qml.RY(inputs[i] * np.pi, wires=i)
    
    # Dilated convolutional layers
    for layer in range(n_layers):
        layer_start = layer * weights_per_layer
        rotation_weights = weights[layer_start:layer_start + n_qubits]
        phase_weights = weights[layer_start + n_qubits:layer_start + 2*n_qubits]
        
        # increasing dilation
        dilation = 2 ** layer
        for i in range(max(1, n_qubits - dilation)):
            qml.CNOT(wires=[i, min(i + dilation, n_qubits-1)])
        
        for i in range(n_qubits):
            position_factor = np.sin((i + 1) / n_qubits * np.pi)
            qml.RY(rotation_weights[i] * position_factor, wires=i)
            qml.PhaseShift(phase_weights[i] * position_factor, wires=i)
    
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]


class HybridQuantumDCNN(nn.Module):
        
    def __init__(self, 
                 input_features: int, 
                 output_features: int,
                 n_qubits: int = 8, 
                 n_layers: int = 3,
                 hidden_dim: int = 64,
                 dropout_rate: float = 0.3):
        
        
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.input_features = input_features
        self.output_features = output_features
        
        # input_features -> n_qubits
        self.feature_reducer = self._build_feature_reducer(
            input_features, n_qubits, hidden_dim
        )
        
        # Quantum layer
        self.qlayer = self._build_quantum_layer(n_qubits, n_layers)

        self.post_processing = self._build_post_processor(
            n_qubits, output_features, hidden_dim, dropout_rate
        )
    
    def _build_feature_reducer(self, input_dim: int, output_dim: int, 
                               hidden_dim: int) -> nn.Module:
        
        if input_dim <= output_dim:
           
            return nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.BatchNorm1d(output_dim),
                nn.ReLU()
            )
        else:
            
            return nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim),
                nn.BatchNorm1d(output_dim),
                nn.ReLU()
            )
    
    def _build_quantum_layer(self, n_qubits: int, n_layers: int) -> nn.Module:
        
        weight_shapes = {"weights": (n_qubits * 2 * n_layers,)}
        dev = qml.device("default.qubit", wires=n_qubits)
        qnode = qml.QNode(
            lambda inputs, weights: quantum_dilated_CNN_circuit(
                inputs, weights, n_qubits, n_layers
            ),
            dev
        )
        
        return qml.qnn.TorchLayer(qnode, weight_shapes)
    
    def _build_post_processor(self, input_dim: int, output_dim: int,
                              hidden_dim: int, dropout_rate: float) -> nn.Module:
        
        mid_dim = min(32, max(16, hidden_dim // 2))
        
        return nn.Sequential(
            nn.Linear(input_dim, mid_dim),
            nn.BatchNorm1d(mid_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mid_dim, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        if len(x.shape) == 3:
            
            batch_size, timesteps, features = x.shape
            x = x.mean(dim=1) 
        else:
            batch_size, features = x.shape
        
        x_reduced = self.feature_reducer(x) 
        
        q_out = torch.stack([
            self.qlayer(x_reduced[i]) for i in range(batch_size)
        ])
        
        
        output = self.post_processing(q_out)
        
        return output
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_qdcnn_model(task_type: str,
                       input_features: int,
                       output_features: int = 1,
                       n_qubits: int = 8,
                       n_layers: int = 3,
                       **kwargs) -> nn.Module:
    
    base_model = HybridQuantumDCNN(
        input_features=input_features,
        output_features=output_features,
        n_qubits=n_qubits,
        n_layers=n_layers,
        **kwargs
    )
    
    
    if task_type == 'regression':
        return base_model
    
    elif task_type == 'binary_classification':
        return nn.Sequential(
            base_model,
            nn.Sigmoid()
        )
    
    elif task_type == 'multi_classification':
        return nn.Sequential(
            base_model,
            nn.Softmax(dim=-1)
        )
    
    else:
        raise ValueError(f"Unknown task_type: {task_type}")