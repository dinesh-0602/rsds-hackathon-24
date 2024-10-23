# Data Driven AI for Remote Sensing

This repository is intended for setups required for IEEE RSDS Hackathon **Data Driven AI for Remote Sensing**. It leverage and AWS SageMaker for building remote sensing AI applications. This README provides a comprehensive guide to get you started with the project setup, training, and evaluation criteria for hackathon. 

## Table of Contents
- [Data Driven AI for Remote Sensing](#data-driven-ai-for-remote-sensing)
  - [Table of Contents](#table-of-contents)
  - [Project Overview](#project-overview)
  - [Getting Started](#getting-started)
  - [Installation Steps after Jupyterlab starts](#installation-steps-after-jupyterlab-starts)
  - [Training Process](#training-process)
  - [Inference details](#inference-details)
  - [Hackathon Evaluation Details](#hackathon-evaluation-details)
    - [IoU Metric Calculation](#iou-metric-calculation)
  - [References](#references)
    - [TerraTorch base repository](#terratorch-base-repository)
    - [Terratorch Quick Start documentation](#terratorch-quick-start-documentation)
    - [albumentations documentation](#albumentations-documentation)
    - [Dataset](#dataset)
  - [Acknowledgements](#acknowledgements)

## Project Overview

This project is part of a hackathon where participants are tasked with developing AI models for remote sensing using AWS SageMaker. Participants will receive a dataset and attend a workshop on training AI foundation models using Jupyter Notebook.

## Getting Started

To participate in the hackathon, you will need to log in to AWS account using the AWS login credentials provided at:

[http://smd-ai-workshop-creds-webapp.s3-website-us-east-1.amazonaws.com/](http://smd-ai-workshop-creds-webapp.s3-website-us-east-1.amazonaws.com/)

Use your assigned team name for login.

![image](https://github.com/user-attachments/assets/7c9634f5-d3cf-4398-bc5f-5ec1ab821202)

use the provided username and password to login 

![image](https://github.com/user-attachments/assets/adc7fdfc-b3f5-4605-99bd-8d5c916b013e)

click Jupyterlab 

![image](https://github.com/user-attachments/assets/5d743902-7556-4a50-b1ef-30c887ed90d9)

Create Jupyterlab space, provide name, and choose "private"

![image](https://github.com/user-attachments/assets/cbd5b10a-5f01-43d1-9450-ab9e2ab85c6c)

choose `ml.g4dn.xlarge` as Instance, set storage to 50GB, click Run Space button.

![image](https://github.com/user-attachments/assets/98448458-1763-4909-bc41-3346e5f7673c)


## Installation Steps after Jupyterlab starts

0. Clone the repository
   ```bash
   git clone https://github.com/NASA-IMPACT/rsds-hackathon-24.git
   ```
   Alternatively, you can click on the `Git` on top left corner and clone the repository, by pasting the URL `https://github.com/NASA-IMPACT/rsds-hackathon-24.git`

1. **Open Terminal and Update and Install System Packages**
   - Open your terminal and run:
     ```bash
     sudo apt-get update && sudo apt-get install ffmpeg libsm6 libxext6 -y
     ```

2. **Install Python Dependencies**
   - Ensure you have Python installed, then install the required packages:
     ```bash
     cd rsds-hackathon-24
     pip install -r requirements.txt
     ```

## Training Process

1. **Run the Training Notebook** 
   - Execute the Jupyter Notebook provided for training. Notebook is [training_terratorch.ipynb](training_terratorch.ipynb).
   - This notebook will:
     - Download the development set of training data.
     - Create necessary directories.
     - Utilize the TerraTorch library to run a sample AI model.
     - Generate results and produce a TensorBoard log for visualization.

2. **Monitor Training with TensorBoard**
   - While training is ongoing, use Weights & Biases (wandb) to sync the TensorBoard file and monitor progress in real-time.

Do the following
```bash
wandb init
```
- click the link to get API key

```bash
cd <path to experiment>
wandb sync <path_to_tensorboard_log_file>
```
> **Note:** Sync needs to be run every time you want to sync the TensorBoard log file to Weights & Biases (wandb).

## Inference details

To run inference using the TerraTorch library, you can use the following command:
terratorch predict -c <path_to_config_file> --ckpt_path<path_to_checkpoint> --predict_output_dir <path_to_output_dir> --data.init_args.predict_data_root <path_to_input_dir>

## Hackathon Evaluation Details 

Participants must provide the following:

1. **Training Notebook**:
   - A Jupyter Notebook to run the model training.
   - Include the trained model weights and necessary logs.
   - Ensure the notebook is easy to run for the judges.

2. **Model Improvement Documentation**:
   - A comprehensive list of attempts to improve model performance.
   - Include results for each attempt.
   - Judges will evaluate the level of effort, decision-making process, and results.

3. **Performance Metrics Calculation**:
   - Calculate Intersection over Union (IoU) as the performance metric.
   - See `inference_terratorch.ipynb` for details on testing the model.

4. **Inference Notebook**:
   - A final notebook to run model inference.
   - The test split will not be provided but will have the same format as the training/validation data.
   - Judges will use this notebook to calculate the IoU score, so ensure all steps are clearly shown.
   - The notebook will be run with a held-out set of data, so do not expect 100% accuracy.

5. **TerraTorch Documentation**:
   - Refer to the [config_explainer.md](configs/config_explainer.md) file for more details. You need to understand the configuration details for potential model improvements.
   - Refer to the TerraTorch [Quick Start](https://ibm.github.io/terratorch/quick_start/) documentation for more details on running model inference and configuration details.

### IoU Metric Calculation

you can use the following formula and Python code snippet for calculating the IoU metric. This will be used for evaluation.

**Formula:**
$$
IoU = \frac{True Positive}{True Positive + False Positive + False Negative}
$$


**Python Code:**
```python
def calculate_iou(y_true, y_pred):
    intersection = np.logical_and(y_true, y_pred)
    union = np.logical_or(y_true, y_pred)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score
```
## References

### TerraTorch base repository
- [TerraTorch](https://github.com/IBM/terratorch)
### Terratorch Quick Start documentation
- [Quick Start](https://ibm.github.io/terratorch/quick_start/)
### albumentations documentation
- [Albumentations](https://albumentations.ai/docs/)
### Dataset
- [HLS Burn Scars dataset](https://huggingface.co/datasets/Muthukumaran/fire_scars_hackathon_dataset)
- [HLS data](https://hls.gsfc.nasa.gov/hls-data/)
- [Burn Scars](https://www.weather.gov/sew/burnscar)

## Acknowledgements

- IBM for providing the TerraTorch library
- NASA-IMPACT for providing the HLS data, code, and instructions
- [IEEE Geoscience and Remote Sensing Society Earth Science Informatics Technical Committee](https://www.grss-ieee.org/technical-committees/earth-science-informatics/) for organizing the hackathon and providing funding for the resources. learn more about the committee [here](https://www.grss-ieee.org/technical-committees/earth-science-informatics/)
- [AWS](https://aws.amazon.com/) for providing the SageMaker resources.