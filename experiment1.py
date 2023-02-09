# %%
import warnings
warnings.filterwarnings('ignore')

# Data manipulation libraries
import os
from pathlib import Path
import zipfile
import requests

import numpy as np
import pandas as pd
import csv

# Data tracking libraries
import wandb
from timeit import default_timer as timer

# Torch libraries
import torch
from torch import nn
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torchvision.models import resnet50, ResNet50_Weights

# %%
# Check torch version
print(f"torch version: {torch.__version__}")
print(f"torchvision version {torchvision.__version__}")

# Check if gpu is available
print(f"Is CUDA available: {torch.cuda.is_available()}")

# Set device agnostic code
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# %%
# List classes you are interested in | Keep it as 6, or it will break
class_names = ['No Finding', 'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']

from custom_dl_functions import download_data
chexpert_folder_path = download_data(source = "http://download.cs.stanford.edu/deep/CheXpert-v1.0-small.zip",
                                    destination = "CheXpert-v1.0-small")

# Get path to csv files
train_csv_path = chexpert_folder_path / 'train.csv'
valid_csv_path = chexpert_folder_path / 'valid.csv'

# %%
# Read CSVs
train_data = pd.read_csv(train_csv_path)
valid_data = pd.read_csv(valid_csv_path)

# Exclude all images that aren't frontal
valid_data = valid_data[valid_data['Path'].str.contains("frontal")]
train_data = train_data[train_data['Path'].str.contains("frontal")]

# Because validation is so small, we need to expand it. Therefore, we take 900 image from training and push it to test
test_data = train_data.head(900)
train_data = train_data[900:]

stack_test_data = pd.concat([valid_data, test_data], axis=0)

# Save augmented csv files (only frontal images)
train_data.to_csv(chexpert_folder_path / 'train_mod.csv', index = False)
stack_test_data.to_csv(chexpert_folder_path / 'stack_mod.csv', index = False)

# Update paths to updated train/test/val files
train_csv_path = chexpert_folder_path / 'train_mod.csv'
stack_test_csv_path = chexpert_folder_path / 'stack_mod.csv'

# Notify user what's happened
print(f"Excluded lateral images...\nTrain set length: {len(train_data)}\nTest set length: {len(stack_test_data)}") #\nValidation set length: {len(valid_data)}")

# %%
from custom_dl_functions import CheXpertDataSet
from custom_dl_functions import count_classes

# Create datasets train, test, validation

# Instenciate the transforms
transformSequence = transforms.Compose([
    transforms.CenterCrop(320),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    ])

# Get the datasets, number of cases in each dataset and inform user on the progress:
# Training Dataset
datasetTrain = CheXpertDataSet(train_csv_path, class_names = class_names, transform = transformSequence, policy = "ones", multilabel = False)
counts_train = count_classes(datasetTrain, class_names)

print("Train dataset length:", len(datasetTrain))
print(f"Where: \n{counts_train} \n")

# Stack_test
datasetStack = CheXpertDataSet(stack_test_csv_path, class_names = class_names, transform = transformSequence, policy = "ones", multilabel = False)
counts_stack = count_classes(datasetStack, class_names)
print("Valid dataset length:", len(datasetStack))
print(f"Where: \n{counts_stack} \n")

# %%
# Create dataloaders
batch_size =  32
num_workers = 0

# Get dataloaders
DataLoaderTrain = DataLoader(dataset = datasetTrain, batch_size = batch_size,
                            shuffle = True, num_workers = num_workers, pin_memory = True)
DataLoaderTest = DataLoader(dataset = datasetStack, batch_size = batch_size,
                            shuffle = True, num_workers = num_workers, pin_memory = True)

# %%
## EXPERIMENTS
# Full information on the planned experiments please visit the following link:
# https://docs.google.com/spreadsheets/d/1qhN6A7OO0meoFdGOWn55ZBCIaM19IsXshSHTipPisX0/edit?usp=sharing

## EXPERIMENT 1 ##
# Goal: Compare impact of freezing all but fully connected layer vs leaving all the learnable parameters unlocked

# Login to wandb
import wandb
wandb.login(key = 'XXXXXXXXXXXXXXXX')

# Make path to save model state dicts to
model_path = Path("Models")
model_path.mkdir(parents = True, exist_ok = True)


# %%
## CASE 0 - Control ##
# Import model
torch.manual_seed(613124)
weights = ResNet50_Weights.DEFAULT
model = torchvision.models.resnet50(weights=weights).to(device)



# Adjust the fully connected layer to output correct number of classes
output_shape = len(class_names)
model.fc = torch.nn.Sequential(
    torch.nn.Dropout(p = 0.2, inplace = True),
    torch.nn.Linear(in_features = 2048,
                    out_features = output_shape,
                    bias = True).to(device)
)

# Set up loss and an optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

# Number of epochs
epochs = 15

# Start the timer
from model_training_engine_chexpert_wandb_v2 import train

# Set up wandb config
run = wandb.init(
        project = "Experiment_1_Chexpert",
        name = "Pretrained_Unfrozen_15E",
        notes = "Compare impact of freezing all but fully connected layer vs leaving all the learnable parameters unlocked",
        config = {
            "Model": "Pytorch's ResNet50",
            "Pretrained": True,
            "Learning Parameters": "Unfrozen",
            "Epochs": epochs,
            "Batch Size": 32,
            "Learning Rate": 0.001,
            "Loss Function": "Cross Entropy Loss",
            "Optimizer": "Adam",
            "Preprocessing": "Transform List A"
            })

# Start timer
start_time = timer()
print("Training commences, timer started...")

# Setup training/testing loop
results = train(model = model,
                train_dataloader = DataLoaderTrain,
                test_dataloader = DataLoaderTest,
                optimizer = optimizer,
                loss_fn = loss_fn,
                epochs = epochs,
                device = device)
end_time = timer()
print(f"[INFO] Total training time: {end_time-start_time: .3f} seconds")
wandb.log({"Total training time [s]": end_time-start_time})

# Save model and upload it to WANDB
model_name = 'Experiment1Case0_15E'

model_save_path = model_path / model_name

torch.save(model.state_dict(), f = model_save_path)
artifact = wandb.Artifact(model_name, type = 'model')
artifact.add_file(model_save_path)
run.log_artifact(artifact)


wandb.finish()
