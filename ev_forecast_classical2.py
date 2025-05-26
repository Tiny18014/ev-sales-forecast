import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time
import os

# Load and preprocess the dataset
df = pd.read_csv("EV_Dataset.csv")  # Adjust path as needed if running locally

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

# Convert to tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

# Create DataLoader
batch_size = 32
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Define model
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

# Use CPU explicitly
device = torch.device("cpu")
model = EVSalesModel(X_tensor.shape[1]).to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# Training loop
num_epochs = 50
loss_history = []

print("Training on CPU...")
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    start_time = time.time()

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    scheduler.step()
    epoch_loss /= len(train_loader)
    loss_history.append(epoch_loss)

    print(f"Epoch {epoch+1}/{num_epochs}: Loss = {epoch_loss:.4f} (Time: {time.time() - start_time:.2f}s)")

# Create models directory if it doesn't exist
os.makedirs("models", exist_ok=True)

# Save the model in the models folder
model_path = os.path.join("models", "enhanced_ev_sales_model.pth")
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")