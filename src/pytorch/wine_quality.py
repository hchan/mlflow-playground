# The data set used in this example is from http://archive.ics.uci.edu/ml/datasets/Wine+Quality
# P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
# Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.

import logging
import sys
import warnings
from urllib.parse import urlparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

import mlflow
import mlflow.pytorch
from mlflow.models import infer_signature

# Set the MLflow tracking URI only if not running in GitHub Actions
if not os.getenv("GITHUB_ACTIONS"):
    mlflow.set_tracking_uri("http://localhost:5000")

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

# Define the neural network model
class WineQualityNet(nn.Module):
    def __init__(self, input_dim):
        super(WineQualityNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Define evaluation metrics
def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

# print the summary of the evaluation metrics
# if we are in GHA, write the summary to the step summary file
def print_summary(rmse, mae, r2):
    summary = f"RMSE: {rmse}\nMAE: {mae}\nR2: {r2}\n"
    print(summary)
    if os.getenv("GITHUB_ACTIONS"):
        with open(os.getenv("GITHUB_STEP_SUMMARY"), "a") as f:
            f.write(summary)

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)
    torch.manual_seed(40)

    # Read the wine-quality csv file from the URL
    csv_url = (
        "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv"
    )
    try:
        data = pd.read_csv(csv_url, sep=";")
    except Exception as e:
        logger.exception(
            "Unable to download training & test CSV, check your internet connection. Error: %s", e
        )

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)

    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["quality"], axis=1).values
    test_x = test.drop(["quality"], axis=1).values
    train_y = train[["quality"]].values
    test_y = test[["quality"]].values

    # Convert to PyTorch tensors
    train_x = torch.tensor(train_x, dtype=torch.float32)
    test_x = torch.tensor(test_x, dtype=torch.float32)
    train_y = torch.tensor(train_y, dtype=torch.float32)
    test_y = torch.tensor(test_y, dtype=torch.float32)

    # Create DataLoader
    train_dataset = TensorDataset(train_x, train_y)
    test_dataset = TensorDataset(test_x, test_y)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Initialize the model, loss function, and optimizer
    input_dim = train_x.shape[1]
    model = WineQualityNet(input_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    def train(model, train_loader, criterion, optimizer, device):
        model.train()
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)  # Move data and target to the correct device
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        return running_loss / len(train_loader)

    # Evaluate the model
    def evaluate(model, test_loader, criterion, device):
        model.eval()
        test_loss = 0.0
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)  # Move data and target to the correct device
                output = model(data)
                loss = criterion(output, target)
                test_loss += loss.item()
                all_preds.extend(output.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        rmse, mae, r2 = eval_metrics(np.array(all_targets), np.array(all_preds))
        return test_loss / len(test_loader), rmse, mae, r2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)  # Move model to the correct device

    # Start an MLflow run
    with mlflow.start_run() as run:
        for epoch in range(1, 6):  # Train for 5 epochs
            train_loss = train(model, train_loader, criterion, optimizer, device)
            test_loss, rmse, mae, r2 = evaluate(model, test_loader, criterion, device)
            
            # Log metrics
            mlflow.log_metric('train_loss', train_loss, step=epoch)
            mlflow.log_metric('test_loss', test_loss, step=epoch)
            mlflow.log_metric('rmse', rmse, step=epoch)
            mlflow.log_metric('mae', mae, step=epoch)
            mlflow.log_metric('r2', r2, step=epoch)

            # Log parameters
            mlflow.log_param('learning_rate', 0.001)
            mlflow.log_param('batch_size', 64)

            # Log model with input example
            input_example = torch.rand(1, input_dim).to(device).detach().cpu().numpy()
            signature = infer_signature(input_example, model(torch.tensor(input_example).to(device)).detach().cpu().numpy())
            mlflow.pytorch.log_model(model, 'model', signature=signature)

            # Save model state dict
            model_path = f"model_epoch_{epoch}.pth"
            torch.save(model.state_dict(), model_path)
            mlflow.log_artifact(model_path)
            os.remove(model_path)

        # Log the final model
        mlflow.pytorch.log_model(model, 'final_model', signature=signature)

        # Log the training script
        mlflow.log_artifact(__file__)

print_summary(rmse, mae, r2)