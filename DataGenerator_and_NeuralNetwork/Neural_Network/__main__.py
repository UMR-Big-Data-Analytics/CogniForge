import os
import csv
import uuid
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

import utils
import dataset as ds
import transformer_encoder_regression as transformer
import metrics

# Setup device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# === Hyperparameters ===
sec_len_output = 1
step_size = 1
sec_len = 202
window_size = sec_len + sec_len_output
batch_size = 256
epochs = 1
learning_rate = 0.000015
early_stop_patience = 50

# === Transformer Parameters ===
num_layers = 2
num_heads = 4
d_model = 512
dropout = 0.1
dim_feedforward = 512
dropouta = 0.2

# === Data Setup ===
target_col_name = "elevation_profile2"
timestamp_col = "Zeit[(ms)]"
exogenous_vars = ["Current[(A)]", "Noise[(V)]", "Drahtrollendrehzahl[(U/min)]", "Voltage[(V)]"]
input_variables = [target_col_name] + exogenous_vars
target_idx = 0

df = pd.read_csv("data/dataset.csv", sep=",", engine="python")

split_ratio = 0.8
split_idx = int(len(df) * split_ratio)
train_data = df.iloc[:split_idx].copy()
test_data = df.iloc[split_idx:].copy()

name_dataset_train = "SplitTrain"
name_dataset_test = "SplitTest"

# Shift elevation profile
train_data.iloc[:, target_idx] = utils.prepare_elevation_profile(sec_len/2, train_data, target_idx)
test_data.iloc[:, target_idx] = utils.prepare_elevation_profile(sec_len/2, test_data, target_idx)

# Scale data
scalers = {}
for var in input_variables:
    scaler = StandardScaler()
    train_data[var] = scaler.fit_transform(train_data[var].values.reshape(-1, 1))
    test_data[var] = scaler.transform(test_data[var].values.reshape(-1, 1))
    scalers[var] = scaler

# Datasets
train_indices = utils.get_indices_entire_sequence(train_data, window_size, step_size)
test_indices = utils.get_indices_entire_sequence(test_data, window_size, step_size)

train_dataset = ds.CustomDataset(torch.tensor(train_data[input_variables].values).float(),
                                 target_feature=target_idx,
                                 indices=train_indices,
                                 windowsize=window_size)
test_dataset = ds.CustomDataset(torch.tensor(test_data[input_variables].values).float(),
                                target_feature=target_idx,
                                indices=test_indices,
                                windowsize=window_size)

train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, drop_last=True)

# Model
model = transformer.TransformerEncoderRegressor(
    num_features=len(exogenous_vars),
    window_size=window_size - 1,
    num_heads=num_heads,
    num_layers=num_layers,
    dropout=dropout,
    d_model=d_model,
    dim_feedforward=dim_feedforward,
    dropouta=dropouta
).to(device)


optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
criterion = torch.nn.MSELoss().to(device)

best_val_loss = float("inf")
epochs_no_improve = 0
loss_values = []

output_dir = model.__class__.__name__ + name_dataset_train
os.makedirs(output_dir, exist_ok=True)

# === Training Loop ===
for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    model.train()
    train_losses = []

    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    avg_train_loss = np.mean(train_losses)

    # Validation
    model.eval()
    val_losses = []
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            val_output = model(X)
            val_loss = criterion(val_output, y)
            val_losses.append(val_loss.item())

    avg_val_loss = np.mean(val_losses)
    scheduler.step(avg_val_loss)
    loss_values.append(avg_val_loss)

    print(f"Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), f"{output_dir}/best_model.pth")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= early_stop_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

# === Evaluation ===
model.load_state_dict(torch.load(f"{output_dir}/best_model.pth"))
model.eval()

all_preds, all_true = [], []

with torch.no_grad():
    for X, y in test_loader:
        X, y = X.to(device), y.to(device)
        pred = model(X).cpu().numpy()
        y = y.cpu().numpy()
        pred = scalers[target_col_name].inverse_transform(pred)
        y = scalers[target_col_name].inverse_transform(y)
        all_preds.append(pred)
        all_true.append(y)

preds_flat = np.concatenate(all_preds).flatten()
true_flat = np.concatenate(all_true).flatten()

# === Save Metrics ===
mae = metrics.mean_absolute_error(preds_flat, true_flat)
mse = metrics.mean_squared_error(preds_flat, true_flat)
rmse = metrics.root_mean_squared_error(preds_flat, true_flat)
r2 = metrics.r_squared(preds_flat, true_flat)

unique_id = str(uuid.uuid4())
metrics_dict = {
    'unique_id': unique_id,
    'model': model.__class__.__name__,
    'mae': mae,
    'mse': mse,
    'rmse': rmse,
    'r2': r2,
    'learning rate': learning_rate,
    'last_epoch': epoch,
    'train_dataset': name_dataset_train,
    'test_dataset': name_dataset_test
}

os.makedirs("metrics_results", exist_ok=True)
with open(f"metrics_results/model_metrics_{name_dataset_train}_{name_dataset_test}.csv", "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=metrics_dict.keys())
    writer.writerow(metrics_dict)

# === Plotting ===
plt.figure(figsize=(10, 6))
plt.plot(loss_values)
plt.title("Validation Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig(f"{output_dir}/val_loss_curve.png")

plt.figure(figsize=(10, 6))
plt.plot(true_flat, label="Actual")
plt.plot(preds_flat, label="Predicted")
plt.legend()
plt.title("Actual vs Predicted")
plt.savefig(f"{output_dir}/predictions_vs_actual_{unique_id}.png")
