
from __future__ import print_function, division
from models.unet.unet_class import Unet
from models.class_dataset import RoadSegDataset
from models.train import train_fn  # Assuming this is the training function
from helper import *
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from torch.optim import lr_scheduler
import torch.utils.data as data
from sklearn.model_selection import KFold
import optuna
import neptune
import neptune.integrations.optuna as optuna_utils

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def objective(trial):
    # Define the hyperparameters to be tuned
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-3)
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'SGD'])
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
    num_epochs = 10  # Fixed for tuning

    # Initialize the model
    model = Unet()  # Assuming Unet() is your model initialization
    model.to(device)

    # Define the optimizer
    if optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    # Assuming you have a function to get your dataloaders
    train_loader, val_loader = get_dataloaders(batch_size)

    # Train the model with the given hyperparameters
    for epoch in range(num_epochs):
        train_fn(train_loader, model, optimizer, device)  # Assuming train_fn is your training function
        # Add validation and tracking code here

    # Evaluate the model and return the metric of interest, e.g., validation accuracy
    val_accuracy = evaluate_model(val_loader, model, device)
    return val_accuracy

# Optuna study
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)  # Number of trials

# Output the best hyperparameters
print(study.best_trial)
