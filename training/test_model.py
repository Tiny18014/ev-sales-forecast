import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Define the same model architecture as in training
class EVSalesModel(nn.Module):
    def __init__(self, input_dim):
        super(EVSalesModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

def load_and_preprocess_data():
    """Load and preprocess the dataset (same as training)"""
    # Try different possible paths for the CSV file
    possible_paths = [
        "EV_Dataset.csv",           # Current directory
        "../EV_Dataset.csv",        # Parent directory
        "../data/EV_Dataset.csv",   # Parent directory data folder
        "data/EV_Dataset.csv"       # Current directory data folder
    ]
    
    df = None
    for path in possible_paths:
        try:
            df = pd.read_csv(path)
            print(f"Dataset loaded from: {path}")
            break
        except FileNotFoundError:
            continue
    
    if df is None:
        raise FileNotFoundError("EV_Dataset.csv not found in any of the expected locations: " + 
                              ", ".join(possible_paths))
    
    # Drop rows with missing values
    df.dropna(inplace=True)
    
    # Encode categorical features
    categorical_cols = df.select_dtypes(include=['object']).columns
    df = pd.get_dummies(df, columns=categorical_cols)
    
    # Apply log transform to target
    df["EV_Sales_Quantity"] = np.log1p(df["EV_Sales_Quantity"])
    
    # Separate features and target
    X = df.drop("EV_Sales_Quantity", axis=1).values
    y = df["EV_Sales_Quantity"].values
    
    # Normalize the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    return X, y, scaler

def evaluate_model(model, X_test, y_test, device):
    """Evaluate the model and return predictions and metrics"""
    model.eval()
    
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_pred = model(X_test_tensor).cpu().numpy()
    
    # Convert back from log scale
    y_test_original = np.expm1(y_test)
    y_pred_original = np.expm1(y_pred.flatten())
    
    # Calculate metrics
    mse = mean_squared_error(y_test_original, y_pred_original)
    mae = mean_absolute_error(y_test_original, y_pred_original)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_original, y_pred_original)
    
    # Calculate percentage error (handle zero values)
    non_zero_mask = y_test_original != 0
    if np.sum(non_zero_mask) > 0:
        mape = np.mean(np.abs((y_test_original[non_zero_mask] - y_pred_original[non_zero_mask]) / y_test_original[non_zero_mask])) * 100
    else:
        mape = float('inf')
    
    return y_pred_original, {
        'MSE': mse,
        'MAE': mae,
        'RMSE': rmse,
        'R2_Score': r2,
        'MAPE': mape
    }

def plot_results(y_true, y_pred, metrics):
    """Create visualizations for model performance"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Actual vs Predicted scatter plot
    axes[0, 0].scatter(y_true, y_pred, alpha=0.6, color='blue')
    axes[0, 0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('Actual EV Sales')
    axes[0, 0].set_ylabel('Predicted EV Sales')
    axes[0, 0].set_title('Actual vs Predicted EV Sales')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Residuals plot
    residuals = y_true - y_pred
    axes[0, 1].scatter(y_pred, residuals, alpha=0.6, color='green')
    axes[0, 1].axhline(y=0, color='r', linestyle='--')
    axes[0, 1].set_xlabel('Predicted EV Sales')
    axes[0, 1].set_ylabel('Residuals')
    axes[0, 1].set_title('Residual Plot')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Distribution of residuals
    axes[1, 0].hist(residuals, bins=30, alpha=0.7, color='orange', edgecolor='black')
    axes[1, 0].set_xlabel('Residuals')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Distribution of Residuals')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Metrics text
    axes[1, 1].axis('off')
    metrics_text = f"""
    Model Performance Metrics:
    
    MSE (Mean Squared Error): {metrics['MSE']:.2f}
    MAE (Mean Absolute Error): {metrics['MAE']:.2f}
    RMSE (Root Mean Squared Error): {metrics['RMSE']:.2f}
    R² Score: {metrics['R2_Score']:.4f}
    MAPE (Mean Absolute Percentage Error): {metrics['MAPE']:.2f}%
    
    R² Score Interpretation:
    - 1.0: Perfect prediction
    - 0.8-1.0: Very good
    - 0.6-0.8: Good
    - 0.4-0.6: Moderate
    - <0.4: Poor
    """
    axes[1, 1].text(0.1, 0.9, metrics_text, transform=axes[1, 1].transAxes, 
                   fontsize=12, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig('model_evaluation_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    print("Loading and preprocessing data...")
    X, y, scaler = load_and_preprocess_data()
    
    # Split data (same random state as training)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Test set size: {len(X_test)} samples")
    
    # Load the trained model
    device = torch.device("cpu")
    model = EVSalesModel(X.shape[1]).to(device)
    
    # Try different possible paths for the model file
    model_paths = [
        "models/enhanced_ev_sales_model.pth",           # Current directory
        "../models/enhanced_ev_sales_model.pth",        # Parent directory
        "enhanced_ev_sales_model.pth"                   # Current directory (if moved)
    ]
    
    model_loaded = False
    for model_path in model_paths:
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Model loaded successfully from: {model_path}")
            model_loaded = True
            break
        except FileNotFoundError:
            continue
        except Exception as e:
            print(f"Error loading model from {model_path}: {e}")
            continue
    
    if not model_loaded:
        print("Error: Model file not found in any of the expected locations:")
        for path in model_paths:
            print(f"  - {path}")
        print("Make sure you've trained the model first.")
        return
    
    # Evaluate the model
    print("Evaluating model on test set...")
    y_pred, metrics = evaluate_model(model, X_test, y_test, device)
    
    # Print results with better formatting
    print("\n" + "="*50)
    print("MODEL EVALUATION RESULTS")
    print("="*50)
    
    for metric_name, value in metrics.items():
        if metric_name == 'R2_Score':
            print(f"{metric_name}: {value:.4f}")
        elif metric_name == 'MAPE':
            if np.isinf(value):
                print(f"{metric_name}: Cannot calculate (too many zero values)")
            else:
                print(f"{metric_name}: {value:.2f}%")
        else:
            print(f"{metric_name}: {value:.2f}")
    
    # Data statistics
    zero_count = np.sum(np.expm1(y_test) == 0)
    total_count = len(y_test)
    print(f"\nData Statistics:")
    print(f"Total test samples: {total_count}")
    print(f"Zero sales records: {zero_count} ({zero_count/total_count*100:.1f}%)")
    print(f"Non-zero sales records: {total_count - zero_count} ({(total_count-zero_count)/total_count*100:.1f}%)")
    
    # Interpretation
    print("\n" + "="*50)
    print("INTERPRETATION")
    print("="*50)
    
    r2 = metrics['R2_Score']
    if r2 >= 0.8:
        interpretation = "Very Good"
    elif r2 >= 0.6:
        interpretation = "Good"
    elif r2 >= 0.4:
        interpretation = "Moderate"
    else:
        interpretation = "Poor"
    
    print(f"Model Performance: {interpretation}")
    print(f"The model explains {r2*100:.1f}% of the variance in EV sales data.")
    print(f"On average, predictions are off by {metrics['MAE']:.0f} units.")
    
    if np.isinf(metrics['MAPE']):
        print(f"MAPE cannot be calculated due to many zero sales values.")
        # Calculate MAPE for non-zero values only
        non_zero_mask = np.expm1(y_test) != 0
        if np.sum(non_zero_mask) > 0:
            non_zero_actual = np.expm1(y_test[non_zero_mask])
            non_zero_predicted = y_pred[non_zero_mask]
            mape_non_zero = np.mean(np.abs((non_zero_actual - non_zero_predicted) / non_zero_actual)) * 100
            print(f"MAPE for non-zero sales only: {mape_non_zero:.1f}%")
    else:
        print(f"Mean Absolute Percentage Error: {metrics['MAPE']:.1f}%")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    plot_results(np.expm1(y_test), y_pred, metrics)
    print("Evaluation complete! Check 'model_evaluation_results.png' for visualizations.")
    
    # Sample predictions
    print("\n" + "="*50)
    print("SAMPLE PREDICTIONS")
    print("="*50)
    print("Actual vs Predicted (first 10 test samples):")
    print("-" * 40)
    
    for i in range(min(10, len(y_test))):
        actual = np.expm1(y_test[i])
        predicted = y_pred[i]
        error = abs(actual - predicted)
        
        if actual == 0:
            error_pct_str = "N/A (zero sales)"
        else:
            error_pct = (error / actual) * 100
            error_pct_str = f"{error_pct:5.1f}%"
        
        print(f"Sample {i+1:2d}: Actual={actual:8.0f}, Predicted={predicted:8.0f}, Error={error:6.0f} ({error_pct_str})")
    
    # Show some non-zero examples if available
    non_zero_indices = np.where(np.expm1(y_test) > 0)[0]
    if len(non_zero_indices) > 0:
        print(f"\nNon-zero sales examples (showing up to 5):")
        print("-" * 40)
        for i, idx in enumerate(non_zero_indices[:5]):
            actual = np.expm1(y_test[idx])
            predicted = y_pred[idx]
            error = abs(actual - predicted)
            error_pct = (error / actual) * 100
            print(f"Sample {idx+1:2d}: Actual={actual:8.0f}, Predicted={predicted:8.0f}, Error={error:6.0f} ({error_pct:5.1f}%)")

if __name__ == "__main__":
    main()