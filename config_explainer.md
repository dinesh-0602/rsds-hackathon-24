## Terratorch Configuration File Explanation

### **General Configuration**

- **lightning.pytorch==2.1.1**: Specifies the version of the PyTorch Lightning library used for training models.

- **seed_everything: 0**: Sets a seed for random number generators to ensure reproducibility.

### **Trainer Configuration**

- **accelerator: auto**: Automatically selects the hardware accelerator (CPU or GPU) for training.

- **strategy: auto**: Chooses the best distributed training strategy automatically.

- **devices: auto**: Determines the number of devices (e.g., GPUs) to use automatically.

- **num_nodes: 1**: Specifies the number of nodes to use in training.

- **precision: 16-mixed**: Uses mixed precision training to speed up computation and reduce memory usage.

#### **Logger**

- **class_path: TensorBoardLogger**: Uses TensorBoard for logging training metrics.
  
  - **init_args**:
    - **save_dir: test**: Directory where logs are saved.
    - **name: fire_scars**: Name of the experiment for logging purposes.

#### **Callbacks**

- **RichProgressBar**: Displays a rich progress bar during training.

- **LearningRateMonitor**:
  - **init_args**:
    - **logging_interval: epoch**: Logs the learning rate at each epoch.

- **EarlyStopping**:
  - **init_args**:
    - **monitor: val/loss**: Monitors validation loss to decide when to stop early.
    - **patience: 40**: Number of epochs with no improvement after which training will be stopped.

#### **Other Trainer Settings**

- **max_epochs: 200**: Maximum number of epochs for training.

- **check_val_every_n_epoch: 1**: Validates the model every epoch.

- **log_every_n_steps: 50**: Logs metrics every 50 steps.

- **enable_checkpointing: true**: Enables model checkpointing.

- **default_root_dir: ./**: Root directory for logs and checkpoints.

### **Data Configuration**

- **class_path: GenericNonGeoSegmentationDataModule**: Specifies the data module class used for loading data.

  - **init_args**:
    - **batch_size: 8**: Number of samples per batch.
    - **num_workers: 8**: Number of workers for data loading.
    - **dataset_bands & output_bands**:
      - Lists spectral bands used in input and output.
    - **rgb_indices**:
      - Indices for RGB channels in input data.
    - **train_transform**:
      - Data augmentation techniques applied during training.
      - Includes `RandomCrop`, `HorizontalFlip`, and `ToTensorV2`.
    - **no_data_replace & no_label_replace**:
      - Values used to replace missing data or labels.
    - Paths for training, validation, and testing datasets.
    - Patterns (`img_grep` and `label_grep`) to identify image and label files.
    - Normalization parameters (`means` and `stds`).
    - **num_classes: 2**: Number of output classes.

### **Model Configuration**

- **class_path: terratorch.tasks.SemanticSegmentationTask**: Defines the task as semantic segmentation.

  - **init_args (model_args)**:
    - Model architecture details like `decoder`, `backbone`, and channel settings.
    - Other parameters include dropout rate and number of convolutions in decoder.
    - Loss function set to `dice`.
    - Additional settings like `plot_on_val`, `ignore_index`, and whether to freeze parts of the model (`freeze_backbone`, `freeze_decoder`).
    - Tiled inference parameters for handling large images during inference.

### **Optimizer Configuration**

- **class_path: torch.optim.Adam**:
  
  - Uses Adam optimizer with specific learning rate (`lr`) and weight decay (`weight_decay`).

### **Learning Rate Scheduler**

- **class_path: ReduceLROnPlateau**:
  
  - Adjusts learning rate based on validation loss improvements.

For further reading, refer to the [PyTorch Lightning Documentation](https://pytorch-lightning.readthedocs.io/en/stable/) and [Albumentations Documentation](https://albumentations.ai/docs/).