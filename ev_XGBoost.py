"""
Quantum Embedding + XGBoost EV Sales Forecasting Model for 30-Day Predictions
=============================================================================
"""

import pennylane as qml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import os
from typing import Tuple, List, Dict, Optional
import time

warnings.filterwarnings('ignore')

# =============== CONFIGURATION ===============
class Config:
    # Quantum Configuration
    N_QUBITS = 4
    N_LAYERS = 2
    QUANTUM_FEATURES = 8  # Number of features to embed quantum
    
    # Training Configuration
    LOOKBACK_DAYS = 60  # Historical window for features
    FORECAST_DAYS = 30  # Prediction horizon
    TRAIN_SPLIT = 0.8
    
    # XGBoost Configuration
    XGB_PARAMS = {
        'objective': 'reg:squarederror',
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 200,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1
    }
    
    # Device Configuration
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    USE_GPU_QUANTUM = True
    DTYPE = torch.float32

config = Config()
print(f"Using device: {config.DEVICE}")

# =============== QUANTUM DEVICE SETUP ===============
def setup_quantum_device():
    """Setup quantum device with GPU support if available"""
    try:
        if config.USE_GPU_QUANTUM:
            dev = qml.device("lightning.gpu", wires=config.N_QUBITS, shots=None)
            print("Using lightning.gpu quantum device")
        else:
            raise Exception("Fallback to CPU")
    except:
        dev = qml.device("lightning.qubit", wires=config.N_QUBITS, shots=None)
        print("Using lightning.qubit quantum device")
    return dev

quantum_dev = setup_quantum_device()

# =============== FEATURE ENGINEERING ===============
class FeatureEngineer:
    """Comprehensive feature engineering for time series forecasting"""
    
    def __init__(self):
        self.scalers = {}
        self.label_encoders = {}
        self.feature_names = []
    
    def create_time_features(self, df: pd.DataFrame, date_col: str = 'Date') -> pd.DataFrame:
        """Create comprehensive time-based features"""
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Basic time features
        df['day_of_week'] = df[date_col].dt.dayofweek
        df['day_of_month'] = df[date_col].dt.day
        df['month'] = df[date_col].dt.month
        df['week_of_year'] = df[date_col].dt.isocalendar().week

        df['quarter'] = df[date_col].dt.quarter
        df['year'] = df[date_col].dt.year
        
        # Binary flags
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_month_start'] = df[date_col].dt.is_month_start.astype(int)
        df['is_month_end'] = df[date_col].dt.is_month_end.astype(int)
        df['is_quarter_start'] = df[date_col].dt.is_quarter_start.astype(int)
        df['is_quarter_end'] = df[date_col].dt.is_quarter_end.astype(int)
        df['is_year_start'] = df[date_col].dt.is_year_start.astype(int)
        df['is_year_end'] = df[date_col].dt.is_year_end.astype(int)
        
        # Cyclical encoding for better ML performance
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_of_year_sin'] = np.sin(2 * np.pi * df[date_col].dt.dayofyear / 365.25)
        df['day_of_year_cos'] = np.cos(2 * np.pi * df[date_col].dt.dayofyear / 365.25)
        
        return df
    
    def create_lag_features(self, df: pd.DataFrame, target_col: str = 'EV_Sales_Quantity') -> pd.DataFrame:
        """Create lag features for temporal memory"""
        df = df.copy()
        
        # Lag features
        for lag in [1, 7, 14, 30]:
            df[f'lag_{lag}'] = df[target_col].shift(lag)
        
        # Difference features
        df['diff_1'] = df[target_col].diff(1)
        df['diff_7'] = df[target_col].diff(7)
        
        # Percentage change features
        df['pct_change_1'] = df[target_col].pct_change(1)
        df['pct_change_7'] = df[target_col].pct_change(7)
        
        return df
    
    def create_rolling_features(self, df: pd.DataFrame, target_col: str = 'EV_Sales_Quantity') -> pd.DataFrame:
        """Create rolling statistics features"""
        df = df.copy()
        
        # Rolling statistics
        for window in [7, 14, 30]:
            df[f'rolling_mean_{window}'] = df[target_col].rolling(window=window).mean()
            df[f'rolling_std_{window}'] = df[target_col].rolling(window=window).std()
            df[f'rolling_median_{window}'] = df[target_col].rolling(window=window).median()
            df[f'rolling_max_{window}'] = df[target_col].rolling(window=window).max()
            df[f'rolling_min_{window}'] = df[target_col].rolling(window=window).min()
        
        # Exponentially weighted moving averages
        for span in [7, 14]:
            df[f'ewm_mean_{span}'] = df[target_col].ewm(span=span).mean()
        
        return df
    
    def create_dynamic_features(self, df: pd.DataFrame, target_col: str = 'EV_Sales_Quantity') -> pd.DataFrame:
        """Create sales dynamics and behavioral features"""
        df = df.copy()
        
        # Cumulative features
        df['cumulative_sales'] = df[target_col].cumsum()
        
        # Growth rate features
        df['sales_growth_rate_7'] = df[target_col].rolling(7).apply(
            lambda x: (x.iloc[-1] - x.iloc[0]) / max(x.iloc[0], 1), raw=False
        )
        
        # Volatility index
        df['volatility_index_7'] = (df['rolling_std_7'] / df['rolling_mean_7']).fillna(0)
        df['volatility_index_14'] = (df['rolling_std_14'] / df['rolling_mean_14']).fillna(0)
        
        # Mean reversion score
        df['mean_reversion_7'] = df[target_col] / df['rolling_mean_7']
        df['mean_reversion_30'] = df[target_col] / df['rolling_mean_30']
        
        # Momentum indicators
        df['momentum_7'] = df[target_col] / df['lag_7'] - 1
        df['momentum_30'] = df[target_col] / df['lag_30'] - 1
        
        return df
    
    def engineer_features(self, df: pd.DataFrame, target_col: str = 'EV_Sales_Quantity') -> pd.DataFrame:
        """Main feature engineering pipeline"""
        print("Starting feature engineering...")
        
        # Sort by date to ensure proper temporal ordering
        if 'Date' in df.columns:
            df = df.sort_values('Date').reset_index(drop=True)
        
        # Create all feature types
        df = self.create_time_features(df)
        df = self.create_lag_features(df, target_col)
        df = self.create_rolling_features(df, target_col)
        df = self.create_dynamic_features(df, target_col)
        
        # Handle categorical variables
        # Handle categorical variables
        categorical_cols = ['State', 'Vehicle_Class', 'Vehicle_Category', 'Vehicle_Type', 'Month_Name']

        for col in categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
        
        # Remove rows with NaN values (due to lags and rolling windows)
        initial_shape = df.shape[0]
        df = df.dropna()
        print(f"Removed {initial_shape - df.shape[0]} rows due to NaN values")
        
        print(f"Feature engineering complete. Final shape: {df.shape}")
        return df

# =============== QUANTUM EMBEDDING LAYER ===============
@qml.qnode(quantum_dev, interface="torch")
def quantum_embedding_circuit(inputs, weights):
    """Quantum circuit for feature embedding"""
    # Angle embedding for input features
    qml.AngleEmbedding(inputs, wires=range(config.N_QUBITS), rotation='Y')
    
    # Variational layers
    for layer in range(config.N_LAYERS):
        # Rotation gates
        for i in range(config.N_QUBITS):
            qml.RY(weights[layer, i, 0], wires=i)
            qml.RZ(weights[layer, i, 1], wires=i)
        
        # Entangling gates
        for i in range(config.N_QUBITS - 1):
            qml.CNOT(wires=[i, i + 1])
        
        # Ring connectivity
        if config.N_QUBITS > 2:
            qml.CNOT(wires=[config.N_QUBITS - 1, 0])
    
    # Measurements
    return [qml.expval(qml.PauliZ(i)) for i in range(config.N_QUBITS)]

class QuantumEmbedding(nn.Module):
    """Quantum embedding layer for feature transformation"""
    
    def __init__(self, n_features: int):
        super().__init__()
        self.n_features = min(n_features, config.QUANTUM_FEATURES)
        self.weights = nn.Parameter(
            torch.randn(config.N_LAYERS, config.N_QUBITS, 2, dtype=config.DTYPE) * 0.1
        )
        
    def forward(self, x):
        """Forward pass through quantum circuit"""
        # Select top features for quantum embedding
        if x.shape[-1] > self.n_features:
            x_quantum = x[:, :self.n_features]
        else:
            x_quantum = x
            
        # Normalize inputs to [-π, π] for angle embedding
        x_quantum = torch.tanh(x_quantum) * np.pi
        
        # Process batch
        if len(x_quantum.shape) == 1:
            x_quantum = x_quantum.unsqueeze(0)
            
        batch_size = x_quantum.shape[0]
        embedded_features = []
        
        for i in range(batch_size):
            # Pad with zeros if needed
            if x_quantum.shape[1] < config.N_QUBITS:
                padded_input = torch.zeros(config.N_QUBITS, dtype=config.DTYPE)
                padded_input[:x_quantum.shape[1]] = x_quantum[i]
            else:
                padded_input = x_quantum[i, :config.N_QUBITS]
                
            embedded = quantum_embedding_circuit(padded_input, self.weights)
            embedded_features.append(torch.stack(embedded))
        
        return torch.stack(embedded_features)

# =============== HYBRID MODEL ===============
class QuantumXGBoostForecaster:
    """Main forecasting model combining quantum embedding with XGBoost"""
    
    def __init__(self):
        self.feature_engineer = FeatureEngineer()
        self.quantum_embedding = None
        self.xgb_models = []  # List of models for multi-step forecasting
        self.scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        self.quantum_feature_names = []
        self.classical_feature_names = []
        
    def select_quantum_features(self, df: pd.DataFrame) -> List[str]:
        """Select features for quantum embedding based on importance and non-linearity"""
        # Priority features for quantum embedding (complex interactions)
        quantum_candidates = [
            'lag_1', 'lag_7', 'lag_14', 'rolling_mean_7', 'rolling_std_7',
            'volatility_index_7', 'sales_growth_rate_7', 'mean_reversion_7',
            'day_of_week_sin', 'day_of_week_cos', 'month_sin', 'month_cos',
            'momentum_7', 'ewm_mean_7', 'diff_1', 'pct_change_7'
        ]
        
        # Select available features
        available_features = [f for f in quantum_candidates if f in df.columns]
        return available_features[:config.QUANTUM_FEATURES]
    
    def prepare_multi_step_targets(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Prepare targets for multi-step forecasting"""
        target_data = df[[target_col]].copy()
        
        # Create future targets
        for step in range(1, config.FORECAST_DAYS + 1):
            target_data[f'{target_col}_t+{step}'] = df[target_col].shift(-step)
        
        # Remove rows with NaN targets
        target_data = target_data.dropna()
        return target_data
    
    def prepare_training_data(self, df: pd.DataFrame, target_col: str = 'EV_Sales_Quantity'):
        """Prepare data for training"""
        print("Preparing training data...")
        
        # Engineer features
        df_features = self.feature_engineer.engineer_features(df, target_col)
        
        # Select quantum features
        self.quantum_feature_names = self.select_quantum_features(df_features)
        print(f"Selected quantum features: {self.quantum_feature_names}")
        
        # Prepare multi-step targets
        target_data = self.prepare_multi_step_targets(df_features, target_col)
        
        # Align features with targets
        min_length = min(len(df_features), len(target_data))
        df_features = df_features.iloc[:min_length]
        target_data = target_data.iloc[:min_length]
        
        # Separate quantum and classical features
        quantum_features = df_features[self.quantum_feature_names].values
        
        # Classical features (exclude target and quantum features)
        exclude_cols = [target_col] + self.quantum_feature_names + ['Date']
        if 'Date' in df_features.columns:
            exclude_cols.append('Date')
        
        classical_cols = [col for col in df_features.columns if col not in exclude_cols]
        self.classical_feature_names = classical_cols
        classical_features = df_features[classical_cols].values
        
        # Scale features
        quantum_features_scaled = self.scaler.fit_transform(quantum_features)
        
        # Scale targets
        target_cols = [f'{target_col}_t+{i}' for i in range(1, config.FORECAST_DAYS + 1)]
        targets = target_data[target_cols].values
        targets_scaled = self.target_scaler.fit_transform(targets)
        
        return quantum_features_scaled, classical_features, targets_scaled, df_features.index[:min_length]
    
    def train(self, df: pd.DataFrame, target_col: str = 'EV_Sales_Quantity'):
        """Train the quantum-classical hybrid model"""
        print("Starting training process...")
        
        # Prepare data
        quantum_features, classical_features, targets, indices = self.prepare_training_data(df, target_col)
        
        # Initialize quantum embedding layer
        self.quantum_embedding = QuantumEmbedding(len(self.quantum_feature_names))
        
        # Convert to tensors
        quantum_tensor = torch.tensor(quantum_features, dtype=config.DTYPE)
        
        # Get quantum embeddings
        print("Computing quantum embeddings...")
        with torch.no_grad():
            quantum_embedded = self.quantum_embedding(quantum_tensor).cpu().numpy()
        
        # Combine quantum and classical features
        if classical_features.shape[1] > 0:
            combined_features = np.concatenate([
                quantum_embedded.reshape(quantum_embedded.shape[0], -1),
                classical_features
            ], axis=1)
        else:
            combined_features = quantum_embedded.reshape(quantum_embedded.shape[0], -1)
        
        # Force numeric dtype (float32) on features and targets
        combined_features = np.array(combined_features, dtype=np.float32)
        targets = np.array(targets, dtype=np.float32)
        
        # Split data
        split_idx = int(len(combined_features) * config.TRAIN_SPLIT)
        X_train, X_val = combined_features[:split_idx], combined_features[split_idx:]
        y_train, y_val = targets[:split_idx], targets[split_idx:]
        
        def check_inf_nan(name, arr):
            arr = np.array(arr, dtype=np.float64)  # ensure float64 for checks
            if np.any(np.isnan(arr)):
                print(f"[WARNING] {name} contains NaNs")
            if np.any(np.isinf(arr)):
                print(f"[WARNING] {name} contains Infs")
            if np.any(np.abs(arr) > 1e6):
                print(f"[WARNING] {name} contains very large values (>1e6)")
        
        # Check for invalid values
        check_inf_nan("X_train", X_train)
        check_inf_nan("X_val", X_val)
        check_inf_nan("y_train", y_train)
        check_inf_nan("y_val", y_val)
        
        # Replace infs with NaNs
        X_train = np.where(np.isinf(X_train), np.nan, X_train)
        X_val = np.where(np.isinf(X_val), np.nan, X_val)
        y_train = np.where(np.isinf(y_train), np.nan, y_train)
        y_val = np.where(np.isinf(y_val), np.nan, y_val)
        
        # Drop rows with NaNs
        mask_train = ~np.isnan(X_train).any(axis=1) & ~np.isnan(y_train).any(axis=1)
        dropped_train = len(X_train) - np.sum(mask_train)
        X_train = X_train[mask_train]
        y_train = y_train[mask_train]
        print(f"Dropped {dropped_train} training rows due to NaN/Inf")
        
        mask_val = ~np.isnan(X_val).any(axis=1) & ~np.isnan(y_val).any(axis=1)
        dropped_val = len(X_val) - np.sum(mask_val)
        X_val = X_val[mask_val]
        y_val = y_val[mask_val]
        print(f"Dropped {dropped_val} validation rows due to NaN/Inf")
        
        # Clip very large values to avoid numerical issues
        X_train = np.clip(X_train, -1e6, 1e6)
        X_val = np.clip(X_val, -1e6, 1e6)
        y_train = np.clip(y_train, -1e6, 1e6)
        y_val = np.clip(y_val, -1e6, 1e6)
        
        # Train XGBoost models for each forecast step
        print(f"Training {config.FORECAST_DAYS} XGBoost models...")
        self.xgb_models = []
        
        for step in range(config.FORECAST_DAYS):
            print(f"Training model for step {step + 1}/{config.FORECAST_DAYS}")
            
            model = xgb.XGBRegressor(**config.XGB_PARAMS)
            model.fit(
                X_train, y_train[:, step],
                eval_set=[(X_val, y_val[:, step])],
                verbose=False
            )
            self.xgb_models.append(model)
        
        # Evaluate on validation set
        val_predictions = self.predict_processed(X_val)
        val_mae = mean_absolute_error(y_val, val_predictions)
        val_rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
        
        print(f"Validation MAE: {val_mae:.4f}")
        print(f"Validation RMSE: {val_rmse:.4f}")
        
        return {
            'val_mae': val_mae,
            'val_rmse': val_rmse,
            'n_features': combined_features.shape[1]
        }

    
    def predict_processed(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on processed features"""
        predictions = []
        for model in self.xgb_models:
            pred = model.predict(X)
            predictions.append(pred)
        return np.column_stack(predictions)
    
    def predict(self, df: pd.DataFrame, target_col: str = 'EV_Sales_Quantity') -> np.ndarray:
        """Make 30-day predictions on new data"""
        # Engineer features using the same pipeline
        df_features = self.feature_engineer.engineer_features(df, target_col)
        
        # Extract quantum and classical features
        quantum_features = df_features[self.quantum_feature_names].values
        classical_features = df_features[self.classical_feature_names].values
        
        # Scale quantum features
        quantum_features_scaled = self.scaler.transform(quantum_features)
        
        # Get quantum embeddings
        quantum_tensor = torch.tensor(quantum_features_scaled, dtype=config.DTYPE)
        with torch.no_grad():
            quantum_embedded = self.quantum_embedding(quantum_tensor).numpy()
        
        # Combine features
        if classical_features.shape[1] > 0:
            combined_features = np.concatenate([
                quantum_embedded.reshape(quantum_embedded.shape[0], -1),
                classical_features
            ], axis=1)
        else:
            combined_features = quantum_embedded.reshape(quantum_embedded.shape[0], -1)
        
        # Make predictions
        predictions_scaled = self.predict_processed(combined_features)
        
        # Inverse transform predictions
        predictions = self.target_scaler.inverse_transform(predictions_scaled)
        
        return predictions
    
    def save_model(self, path: str):
        """Save the complete model"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        model_data = {
            'quantum_weights': self.quantum_embedding.state_dict(),
            'xgb_models': self.xgb_models,
            'scaler': self.scaler,
            'target_scaler': self.target_scaler,
            'feature_engineer': self.feature_engineer,
            'quantum_feature_names': self.quantum_feature_names,
            'classical_feature_names': self.classical_feature_names
        }
        
        joblib.dump(model_data, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load the complete model"""
        model_data = joblib.load(path)
        
        # Restore quantum embedding
        self.quantum_embedding = QuantumEmbedding(len(model_data['quantum_feature_names']))
        self.quantum_embedding.load_state_dict(model_data['quantum_weights'])
        
        # Restore other components
        self.xgb_models = model_data['xgb_models']
        self.scaler = model_data['scaler']
        self.target_scaler = model_data['target_scaler']
        self.feature_engineer = model_data['feature_engineer']
        self.quantum_feature_names = model_data['quantum_feature_names']
        self.classical_feature_names = model_data['classical_feature_names']
        
        print(f"Model loaded from {path}")

# =============== EVALUATION AND VISUALIZATION ===============
def evaluate_forecasts(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """Comprehensive evaluation of multi-step forecasts"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Step-wise evaluation
    step_mae = [mean_absolute_error(y_true[:, i], y_pred[:, i]) for i in range(y_true.shape[1])]
    step_rmse = [np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i])) for i in range(y_true.shape[1])]
    
    return {
        'overall_mae': mae,
        'overall_rmse': rmse,
        'step_mae': step_mae,
        'step_rmse': step_rmse
    }

def plot_forecast_results(y_true: np.ndarray, y_pred: np.ndarray, save_path: str = None):
    """Plot comprehensive forecast results"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Overall predictions vs actual
    axes[0, 0].scatter(y_true.flatten(), y_pred.flatten(), alpha=0.5)
    min_val, max_val = min(y_true.flatten()), max(y_true.flatten())
    axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--')
    axes[0, 0].set_xlabel('Actual')
    axes[0, 0].set_ylabel('Predicted')
    axes[0, 0].set_title('Overall Predictions vs Actual')
    axes[0, 0].grid(True)
    
    # Sample forecast trajectories
    sample_indices = np.random.choice(len(y_true), min(5, len(y_true)), replace=False)
    for idx in sample_indices:
        axes[0, 1].plot(range(1, y_true.shape[1] + 1), y_true[idx], 'o-', alpha=0.7, label=f'Actual {idx}')
        axes[0, 1].plot(range(1, y_pred.shape[1] + 1), y_pred[idx], 's--', alpha=0.7, label=f'Pred {idx}')
    axes[0, 1].set_xlabel('Forecast Horizon (Days)')
    axes[0, 1].set_ylabel('Sales')
    axes[0, 1].set_title('Sample Forecast Trajectories')
    axes[0, 1].grid(True)
    axes[0, 1].legend()
    
    # Step-wise MAE
    step_mae = [mean_absolute_error(y_true[:, i], y_pred[:, i]) for i in range(y_true.shape[1])]
    axes[1, 0].plot(range(1, len(step_mae) + 1), step_mae, 'bo-')
    axes[1, 0].set_xlabel('Forecast Horizon (Days)')
    axes[1, 0].set_ylabel('MAE')
    axes[1, 0].set_title('MAE by Forecast Horizon')
    axes[1, 0].grid(True)
    
    # Residuals distribution
    residuals = (y_pred - y_true).flatten()
    axes[1, 1].hist(residuals, bins=50, alpha=0.7)
    axes[1, 1].set_xlabel('Residuals')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Residuals Distribution')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Plots saved to {save_path}")
    
    plt.show()

# =============== MAIN EXECUTION ===============
def main():
    """Main execution function"""
    print("Quantum Embedding + XGBoost EV Sales Forecaster")
    print("=" * 50)
    
    # Load data
    data_path = "EV_Dataset.csv"
    print(f"Loading data from {data_path}...")
    
    try:
        df = pd.read_csv(data_path)
        print(f"Data loaded successfully. Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
    except FileNotFoundError:
        print(f"Error: Could not find {data_path}")
        return
    
    # Initialize and train model
    forecaster = QuantumXGBoostForecaster()
    
    start_time = time.time()
    training_results = forecaster.train(df)
    training_time = time.time() - start_time
    
    print(f"\nTraining completed in {training_time:.2f} seconds")
    print(f"Number of features: {training_results['n_features']}")
    
    # Save model
    model_path = "models/quantum_xgboost_forecaster.pkl"
    forecaster.save_model(model_path)
    
    # Example prediction (using last part of training data)
    print("\nMaking example predictions...")
    sample_predictions = forecaster.predict(df.tail(100))
    
    print(f"Sample predictions shape: {sample_predictions.shape}")
    print(f"First prediction (30-day forecast): {sample_predictions[0]}")
    
    print("\nModel training and evaluation complete!")
    print(f"Model saved to: {model_path}")

if __name__ == "__main__":
    main()