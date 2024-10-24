import torch
import numpy as np
from torch.utils.data import DataLoader
from terratorch.data.datamodule import GenericNonGeoSegmentationDataModule
from terratorch.tasks.segmentation import SemanticSegmentationTask
from pytorch_lightning import Trainer
import yaml

def calculate_iou(y_true, y_pred):
    intersection = np.logical_and(y_true, y_pred)
    union = np.logical_or(y_true, y_pred)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score

# Load the configuration file
with open('configs/fire_scars.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Initialize the data module
data_module = GenericNonGeoSegmentationDataModule(**config['data']['init_args'])
data_module.setup()

# Initialize the model
model = SemanticSegmentationTask(**config['model']['init_args'])

# Load the model checkpoint
checkpoint_path = 'logs/fire_scars/version_3/checkpoints'
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['state_dict'])
model.eval()

# Initialize the trainer
trainer = Trainer(**config['trainer'])

# Get validation data loader
validation_loader = DataLoader(data_module.val_dataloader(), batch_size=1, shuffle=False)

# Make predictions
all_y_true = []
all_y_pred = []

model.eval()
with torch.no_grad():
    for batch in validation_loader:
        inputs, labels = batch
        outputs = model(inputs)
        
        # Apply a threshold to get binary predictions (adjust threshold as needed)
        preds = outputs > 0.5
        
        all_y_true.append(labels.numpy())
        all_y_pred.append(preds.numpy())

# Convert lists to numpy arrays
all_y_true = np.concatenate(all_y_true, axis=0)
all_y_pred = np.concatenate(all_y_pred, axis=0)

# Calculate IoU
iou_score = calculate_iou(all_y_true, all_y_pred)
print(f'IoU Score: {iou_score}')
