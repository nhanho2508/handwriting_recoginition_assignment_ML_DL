﻿# Handwriting Recognition with IAM Dataset

This project focuses on building a deep learning model for **offline handwriting recognition** using the **IAM Handwriting Dataset**. The model is capable of recognizing and transcribing handwritten words from input images.

## Dataset

The dataset used is the **IAM Handwriting Dataset**, which contains labeled images of handwritten English words.

- Downloaded and extracted from: https://www.kaggle.com/datasets/naderabdalghani/iam-handwritten-forms-dataset

## Model Architecture

![Model overal architecture](assets/Model.drawio%20(2).png)

The architecture consists of:

- A convolutional backbone using **residual blocks** to extract spatial features.
![Description ResBlock](assets/Res_Block.drawio.png)
- A **Bidirectional LSTM** layer to capture sequential dependencies.
- A **CTC (Connectionist Temporal Classification)** loss function to enable transcription of variable-length outputs.

The model is built and compiled using TensorFlow and Keras.

## Project Structure

- `train.py`: Train the model on IAM dataset with data augmentation and callbacks (e.g., early stopping, model checkpointing, learning rate scheduler).
- `test.py`: Evaluate the trained model on validation data using Character Error Rate (CER).
- `model.py`: Defines the neural network architecture using residual blocks and BLSTM.
- `preprocessing.py`: Handles dataset parsing, preprocessing, transformation, and data pipeline creation.
- `utils.py`: Utility functions, model configurations, model inference class, and helpers.

## Training

To start training:

```bash
python train.py
```
## Key Configurations

- **Input image size**: `128x32`
- **Batch size**: `16`
- **Optimizer**: `Adam`
- **Loss function**: `CTC`
- **Epochs**: `1000`
- **Data augmentation**:
  - Brightness adjustment
  - Rotation
  - Erosion/Dilation
  - Sharpening

## Inference

Run the model on validation images:

```bash
python test.py
```

## Inference

This script will:

- Load the trained **ONNX model**
- Perform inference on **validation images**
- Calculate and display the **Character Error Rate (CER)**

## Requirements

- Python `3.7+`
- `TensorFlow`
- `OpenCV`
- `pandas`
- `tqdm`
- [`mltu`](https://pypi.org/project/mltu/) library


