"""
EV Sales Prediction Script
=========================
This script demonstrates how to use the trained hybrid model to make predictions
on new data.
"""

import torch
import joblib
import pandas as pd
import numpy as np
import os
from ev_forecast_implementation import load_model, predict


def main():
    
    """
    Example prediction script
    """
    # File paths
    model_dir = "models"
    model_path = os.path.join(model_dir, "ev_sales_hybrid_model.pth")
    norm_params_path = os.path.join(model_dir, "normalization_params.pkl")
    
    # Check if model exists
    if not os.path.exists(model_path) or not os.path.exists(norm_params_path):
        print("Error: Model files not found. Please train the model first.")
        return
    
    # Load normalization parameters to get input features
    norm_params = joblib.load(norm_params_path)
    
    # Get input features
    feature_names = norm_params['feature_names']
    print(f"Required input features: {feature_names}")
    
    # Load the model
    input_dim = len(feature_names)
    model = load_model(model_path, input_dim)
    
    # Example: Create sample input data (adjust this for your actual use case)
    # Option 1: Create a sample dataframe with the same features
    sample_data = create_sample_input(feature_names)
    
    # Option 2: Load data from a CSV file
    # sample_data = pd.read_csv("new_data.csv")
    
    # Make predictions
    predictions = predict(model, sample_data, norm_params)
    
    # Display results
    print("\nPredictions:")
    for i, pred in enumerate(predictions):
        print(f"Sample {i+1}: Predicted EV Sales = {pred:.2f}")


def create_sample_input(feature_names):
    """
    Create sample input data for demonstration
    """
    # Create a dictionary with default values
    data = {}
    
    # Handle common features
    for feature in feature_names:
        if feature == "Year":
            data[feature] = [0.8]  # Normalized value for recent year
        elif feature.startswith("State_"):
            # Set all states to 0 except one
            data[feature] = [1 if feature == "State_California" else 0]
        else:
            # Default value for other features
            data[feature] = [0.5]
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    print("Created sample input with the following values:")
    print(df)
    
    return df


if __name__ == "__main__":
    main()