# Data Driven AI for Remote Sensing

This repository is intended for setups required for RSDS Hackathon **Data Driven AI for Remote Sensing**. It leverage and AWS SageMaker for building remote sensing AI applications. This README provides a comprehensive guide to get you started with the project setup, training, and evaluation criteria for hackathon.

## Table of Contents
- [Data Driven AI for Remote Sensing](#data-driven-ai-for-remote-sensing)
  - [Table of Contents](#table-of-contents)
  - [Project Overview](#project-overview)
  - [Getting Started](#getting-started)
  - [Installation Steps after Jupyterlab starts](#installation-steps-after-jupyterlab-starts)
  - [Training Process](#training-process)
  - [Monitoring and Evaluation](#monitoring-and-evaluation)
    - [IoU Metric Calculation](#iou-metric-calculation)

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

1. **Update and Install System Packages**
   - Open your terminal and run:
     ```bash
     sudo apt-get update && sudo apt-get install ffmpeg libsm6 libxext6 -y
     ```

2. **Install Python Dependencies**
   - Ensure you have Python installed, then install the required packages:
     ```bash
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

## Monitoring and Evaluation

Participants are required to provide a notebook that demonstrates how to:
- Run the trained model.
- Retrieve data from Hugging Face datasets.
- Calculate performance metrics, specifically Intersection over Union (IoU).

### IoU Metric Calculation

For 2D multiband data, you can use the following formula and Python code snippet:

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
