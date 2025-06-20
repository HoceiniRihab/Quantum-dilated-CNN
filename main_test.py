

import torch
import matplotlib.pyplot as plt
import numpy as np


from data_adding_problem import load_adding_problem_data, get_adding_problem_info
from data_music import load_nottingham_data, get_music_info
from quantum_model import (
    train_quantum_model, 
    evaluate_model, 
    plot_training_curves,
    get_model_info
)

def print_banner(text):
    
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60)

def test_adding_problem():
    
    print_banner("ADDING PROBLEM TEST")
    
    
    info = get_adding_problem_info()
    print(f"Task: {info['task_name']}")
    print(f"Description: {info['description']}")
    print(f"Sequence Length: {info['sequence_length']}")
    X_train, X_valid, X_test, y_train, y_valid, y_test = load_adding_problem_data(
        n_samples=1000, seq_length=600
    )
    
    model, train_losses, val_losses = train_quantum_model(
        X_train, X_valid, y_train, y_valid, 
        epochs=100, learning_rate=0.001, 
        task_name="Adding Problem"
    )
    
    test_loss = evaluate_model(model, X_test, y_test, "Adding Problem")
    plot_training_curves(train_losses, val_losses, "Adding Problem")
    
    return model, test_loss

def test_music_modeling():
    
    print_banner("MUSIC MODELING TEST") 
    info = get_music_info()
    print(f"Task: {info['task_name']}")
    print(f"Description: {info['description']}")
    print(f"Data Source: {info['data_source']}")
    
    
   
    X_train, X_valid, X_test = load_nottingham_data()
    model, train_losses, val_losses = train_quantum_model(
        X_train, X_valid, None, None,
        epochs=100, learning_rate=0.001,
        task_name="Music Modeling"
    )
    
    
    test_outputs = evaluate_model(model, X_test, None, "Music Modeling")
    plot_training_curves(train_losses, val_losses, "Music Modeling")
    
    return model, test_outputs

def compare_results(adding_loss, music_outputs):
    
    print_banner("RESULTS COMPARISON")
    
    model_info = get_model_info()
    print(f"Model: {model_info['model_name']}")
    print(f"Architecture: {model_info['n_qubits']} qubits, {model_info['n_layers']} layers")
    print(f"Features: {', '.join(model_info['circuit_features'])}")
    print(f"\nPerformance Summary:")
    print(f"Adding Problem - Test Loss: {adding_loss:.4f}")
    print(f"Music Modeling - Successfully processed {len(music_outputs) if hasattr(music_outputs, '__len__') else 'N/A'} test samples")
    
    

def main():
    
    print_banner("QUANTUM Dilated CNN")
    
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    try:
        # Test 1: adding problem
        print("\n Starting Adding Problem test...")
        adding_model, adding_loss = test_adding_problem()
        # Test 2: music   
        print("\n Starting Music Modeling test..")
        music_model, music_outputs = test_music_modeling()
       
        compare_results(adding_loss, music_outputs)
        
        return {
            'adding_model': adding_model,
            'music_model': music_model,
            'adding_loss': adding_loss,
            'music_outputs': music_outputs
        }
        
    except Exception as e:
        print(f"\n Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    
    torch.manual_seed(42)
    np.random.seed(42)
    results = main()
    
    if results:
        print("\n Testing completed successfully!")
        print("Models are ready for further experimentation")
    else:
        print("\n Testing failed")