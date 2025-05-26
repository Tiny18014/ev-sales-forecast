"""
Simplified EV Sales Forecasting Model
====================================
A streamlined implementation for faster training with reduced computational requirements.
"""

import pennylane as qml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, r2_score
import pandas as pd
import os
import joblib
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

# ================= CONFIGURATION =================
# Adjust these parameters based on available resources
REDUCED_EPOCHS = 20        # Significantly reduced from 200
USE_FLOAT32 = True         # Use single precision instead of double
SIMPLIFIED_QUANTUM = True  # Use simpler quantum circuit
BATCH_SIZE = 16            # Smaller batch size
SAVE_MODEL = True          # Save model after training
SHOTS = 100                # Reduced shots for quantum simulation (faster but less precise)

# ================= DATA PREPROCESSING =================

def preprocess_data(csv_path):
    """
    Preprocess the EV dataset for model training with option for reduced complexity
    """
    print("Loading and preprocessing data...")
    
    # Load dataset
    df = pd.read_csv(csv_path)
    
    # Drop unnecessary categorical columns
    df = df.drop(columns=["Month_Name", "Date", "Vehicle_Class", "Vehicle_Category", "Vehicle_Type"], errors="ignore")
    
    # One-hot encode "State" column
    df = pd.get_dummies(df, columns=["State"], drop_first=True)
    
    # Normalize "Year" column
    df["Year"] = (df["Year"] - df["Year"].min()) / (df["Year"].max() - df["Year"].min())
    
    # Store normalization parameters for later use
    y_mean = df["EV_Sales_Quantity"].mean()
    y_std = df["EV_Sales_Quantity"].std()
    
    # Normalize "EV_Sales_Quantity"
    df["EV_Sales_Quantity"] = (df["EV_Sales_Quantity"] - y_mean) / y_std
    
    # Convert all columns to numeric (force conversion)
    df = df.apply(pd.to_numeric, errors="coerce")
    
    # Drop any remaining NaN values
    df = df.dropna()
    
    # Extract features and target
    dtype = np.float32 if USE_FLOAT32 else np.float64
    torch_dtype = torch.float32 if USE_FLOAT32 else torch.float64
    
    X = df.drop(columns=["EV_Sales_Quantity"]).values.astype(dtype)
    y = df["EV_Sales_Quantity"].values.astype(dtype)
    
    # Get column names for feature reference
    feature_names = df.drop(columns=["EV_Sales_Quantity"]).columns.tolist()
    
    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X, dtype=torch_dtype)
    y_tensor = torch.tensor(y, dtype=torch_dtype).reshape(-1, 1)
    
    # Save normalization parameters
    norm_params = {
        'y_mean': y_mean,
        'y_std': y_std,
        'feature_names': feature_names
    }
    
    print(f"Data preprocessing complete. Dataset shape: {X_tensor.shape}")
    
    return X_tensor, y_tensor, norm_params


# ================= MODEL DEFINITION =================

# Define quantum device with configurable parameters
dev = qml.device("default.qubit", wires=2, shots=SHOTS if SIMPLIFIED_QUANTUM else None)

# Quantum Model - simplified if needed
@qml.qnode(dev, interface="torch")
def quantum_model(inputs, weights):
    """Quantum circuit definition"""
    qml.AngleEmbedding(inputs, wires=[0, 1])
    
    if SIMPLIFIED_QUANTUM:
        # Use just a single entangling layer
        qml.StronglyEntanglingLayers(weights[0:1], wires=[0, 1])
    else:
        # Use full entangling layers
        qml.StronglyEntanglingLayers(weights, wires=[0, 1])
        
    return qml.expval(qml.PauliZ(0))


# Classical Neural Network (Feature Extractor)
class ClassicalNN(nn.Module):
    def __init__(self, input_dim):
        super(ClassicalNN, self).__init__()
        
        torch_dtype = torch.float32 if USE_FLOAT32 else torch.float64
        hidden_size = 8 if SIMPLIFIED_QUANTUM else 16  # Smaller hidden layer if simplified
        
        self.fc1 = nn.Linear(input_dim, hidden_size, dtype=torch_dtype)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 2, dtype=torch_dtype)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class HybridModel(nn.Module):
    def __init__(self, input_dim):
        super(HybridModel, self).__init__()
        self.classical_nn = ClassicalNN(input_dim)
        
        torch_dtype = torch.float32 if USE_FLOAT32 else torch.float64
        weight_layers = 1 if SIMPLIFIED_QUANTUM else 3  # Fewer weight layers if simplified
        
        self.quantum_weights = nn.Parameter(torch.randn((weight_layers, 2, 3), dtype=torch_dtype))

    def forward(self, x):
        x_classical = self.classical_nn(x)
        return quantum_model(x_classical, self.quantum_weights)


# ================= TRAINING FUNCTION =================

def train_model(X_tensor, y_tensor, input_dim, model_path=None):
    """Train the hybrid model with current configuration"""
    print(f"\nTraining model with configuration:")
    print(f"- Epochs: {REDUCED_EPOCHS}")
    print(f"- Precision: {'float32' if USE_FLOAT32 else 'float64'}")
    print(f"- Quantum circuit: {'simplified' if SIMPLIFIED_QUANTUM else 'full'}")
    print(f"- Batch size: {BATCH_SIZE}")
    
    # Initialize model
    hybrid_model = HybridModel(input_dim)
    
    # Setup optimizer and loss function
    learning_rate = 0.01 if SIMPLIFIED_QUANTUM else 0.005
    optimizer = optim.Adam(hybrid_model.parameters(), lr=learning_rate)
    loss_function = nn.MSELoss()
    
    # Create DataLoader
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Training history
    loss_history = []
    
    # Training Loop
    print(f"\nStarting training for {REDUCED_EPOCHS} epochs...")
    
    for epoch in range(REDUCED_EPOCHS):
        epoch_start_time = time.time()
        total_loss = 0
        
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            
            # Forward pass
            batch_pred = torch.zeros(batch_y.size(), 
                                    dtype=torch.float32 if USE_FLOAT32 else torch.float64)
            
            for i in range(len(batch_x)):
                batch_pred[i] = hybrid_model(batch_x[i])
            
            # Calculate loss
            loss = loss_function(batch_pred, batch_y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * batch_x.size(0)
        
        # Average loss for the epoch
        avg_loss = total_loss / len(X_tensor)
        loss_history.append(avg_loss)
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        
        # Print progress for every epoch
        print(f"Epoch {epoch+1}/{REDUCED_EPOCHS}: Loss = {avg_loss:.4f} (Time: {epoch_time:.2f}s)")
    
    print("\nTraining complete!")
    
    # Save the model if requested
    if SAVE_MODEL and model_path:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(hybrid_model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
    
    return hybrid_model, loss_history


# ================= EVALUATION FUNCTIONS =================

def evaluate_model(model, X_tensor, y_tensor, norm_params):
    """Evaluate model performance"""
    print("\nEvaluating model performance...")
    
    # Limit number of samples to evaluate to save time
    eval_samples = min(len(X_tensor), 100)  # Evaluate on at most 100 samples
    
    # Switch to evaluation mode
    model.eval()
    
    # Get predictions
    with torch.no_grad():
        y_pred_list = [model(X_tensor[i]).item() for i in range(eval_samples)]
    
    y_actual = y_tensor[:eval_samples].numpy().flatten()
    
    # Calculate metrics
    mse = np.mean((np.array(y_pred_list) - y_actual) ** 2)
    mae = mean_absolute_error(y_actual, y_pred_list)
    r2 = r2_score(y_actual, y_pred_list)
    
    # Accuracy within threshold
    accuracy = np.mean(np.abs(np.array(y_pred_list) - y_actual) < 0.1) * 100
    
    # Print results
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"R² Score: {r2:.4f}")
    print(f"Accuracy (within normalized ±0.1 error): {accuracy:.2f}%")
    
    # Print sample predictions
    sample_size = min(5, eval_samples)
    print(f"\nSample predictions (first {sample_size}):")
    print("Predicted (normalized):", np.array(y_pred_list[:sample_size]))
    print("Actual (normalized):", y_actual[:sample_size])
    
    # Denormalize to original scale
    y_pred_denorm = np.array(y_pred_list) * norm_params['y_std'] + norm_params['y_mean']
    y_actual_denorm = y_actual * norm_params['y_std'] + norm_params['y_mean']
    
    print("\nAfter denormalization:")
    print("Predicted:", y_pred_denorm[:sample_size])
    print("Actual:", y_actual_denorm[:sample_size])
    
    return {
        'mse': mse,
        'mae': mae,
        'r2': r2,
        'accuracy': accuracy,
        'predictions': y_pred_list,
        'actual': y_actual
    }


def plot_results(metrics, loss_history, save_path=None):
    """Plot training loss and prediction results"""
    plt.figure(figsize=(12, 5))
    
    # Plot training loss
    plt.subplot(1, 2, 1)
    plt.plot(loss_history)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    # Plot predictions vs actual
    plt.subplot(1, 2, 2)
    plt.scatter(metrics['actual'], metrics['predictions'], alpha=0.5)
    min_val = min(min(metrics['actual']), min(metrics['predictions']))
    max_val = max(max(metrics['actual']), max(metrics['predictions']))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    plt.title('Actual vs Predicted (Normalized)')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    
    plt.show()


# ================= MODEL SAVING AND LOADING =================

def save_model_and_params(model, norm_params, model_path, params_path):
    """Save model and normalization parameters"""
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    joblib.dump(norm_params, params_path)
    print(f"Model saved to {model_path}")
    print(f"Parameters saved to {params_path}")


def load_model(model_path, input_dim):
    """Load a saved model"""
    print(f"Loading model from {model_path}...")
    
    # Initialize model with correct dimensions
    model = HybridModel(input_dim)
    
    # Load weights
    model.load_state_dict(torch.load(model_path))
    
    # Set to evaluation mode
    model.eval()
    
    return model


def predict(model, input_data, norm_params):
    """Make predictions with the model"""
    # Prepare input data
    required_features = norm_params['feature_names']
    
    # Check for missing columns
    missing_cols = set(required_features) - set(input_data.columns)
    if missing_cols:
        raise ValueError(f"Input data is missing required columns: {missing_cols}")
    
    # Select required columns
    input_data = input_data[required_features]
    
    # Convert to tensor with appropriate precision
    dtype = torch.float32 if USE_FLOAT32 else torch.float64
    X = torch.tensor(input_data.values.astype(np.float32 if USE_FLOAT32 else np.float64), dtype=dtype)
    
    # Make predictions
    model.eval()
    with torch.no_grad():
        predictions_norm = [model(X[i]).item() for i in range(len(X))]
    
    # Denormalize predictions
    predictions = np.array(predictions_norm) * norm_params['y_std'] + norm_params['y_mean']
    
    return predictions


# ================= MAIN EXECUTION =================

import time

def main():
    """Main execution function"""
    # Track total execution time
    start_time = time.time()
    
    # File paths
    data_path = "EV_Dataset.csv"
    model_dir = "models"
    model_path = os.path.join(model_dir, "ev_sales_hybrid_model_simple.pth")
    norm_params_path = os.path.join(model_dir, "normalization_params.pkl")
    plot_path = os.path.join(model_dir, "training_results.png")
    
    # Create directory for model
    os.makedirs(model_dir, exist_ok=True)
    
    # Load and preprocess data
    X_tensor, y_tensor, norm_params = preprocess_data(data_path)
    
    # Get input dimension
    input_dim = X_tensor.shape[1]
    
    # Train model
    model, loss_history = train_model(X_tensor, y_tensor, input_dim, model_path)
    
    # Save normalization parameters
    joblib.dump(norm_params, norm_params_path)
    print(f"Normalization parameters saved to {norm_params_path}")
    
    # Evaluate model
    metrics = evaluate_model(model, X_tensor, y_tensor, norm_params)
    
    # Plot and save results
    plot_results(metrics, loss_history, plot_path)
    
    # Calculate total execution time
    total_time = time.time() - start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    
    print("\nTo use this model for predictions:")
    print("1. Load the model: model = load_model('models/ev_sales_hybrid_model_simple.pth', input_dim)")
    print("2. Load params: norm_params = joblib.load('models/normalization_params.pkl')")
    print("3. Make predictions: predictions = predict(model, input_data, norm_params)")


if __name__ == "__main__":
    main()