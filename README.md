# Feline-Canine Classifier

This repository contains a Jupyter notebook (`code.ipynb`) for training a deep learning model to classify images of felines and canines. The model is built using PyTorch and leverages a pre-trained VGG-11 network.

## Table of Contents
- [Installation](#installation)
- [Dataset](#dataset)
- [Notebook Overview](#notebook-overview)
- [Training](#training)
- [Evaluation](#evaluation)
- [Submission](#submission)
- [Acknowledgements](#acknowledgements)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/Feline-Canine-Classifier.git
    cd Feline-Canine-Classifier
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Dataset

The dataset used in this project consists of images of felines and canines. The dataset is divided into training and validation sets, and a separate Kaggle test set is used for final evaluation.

- **Training Set**: Contains labeled images for training the model.
- **Validation Set**: Contains labeled images for validating the model during training.
- **Kaggle Test Set**: Contains images for final evaluation and submission to Kaggle.

## Notebook Overview

The `code.ipynb` notebook is structured as follows:

1. **Imports and Setup**: Import necessary libraries and set up the environment.
2. **Data Loading and Preprocessing**: Download and preprocess the dataset.
3. **Model Definition**: Define the VGG-11 based classifier model.
4. **Training**: Train the model on the training dataset.
5. **Evaluation**: Evaluate the model on the validation dataset.
6. **Submission**: Generate predictions for the Kaggle test set and save them for submission.

## Training

The training process involves the following steps:

1. **Set Seed**: Ensure reproducibility by setting a random seed.
2. **Load Pre-trained Model**: Load a pre-trained VGG-11 model and freeze its parameters.
3. **Define Classifier**: Replace the classifier part of the VGG-11 model with a custom classifier.
4. **Train Model**: Train the model using the training dataset and validate it using the validation dataset.
5. **Save Best Checkpoint**: Save the model checkpoint with the best validation accuracy.

## Evaluation

The evaluation process involves:

1. **Load Best Checkpoint**: Load the best model checkpoint saved during training.
2. **Evaluate on Validation Set**: Evaluate the model on the validation dataset to check its performance.
3. **Generate Predictions**: Generate predictions for the Kaggle test set.

## Submission

To generate the submission file for Kaggle:

1. **Load Kaggle Test Set**: Load and preprocess the Kaggle test set.
2. **Generate Predictions**: Use the trained model to generate predictions for the test set.
3. **Save Predictions**: Save the predictions in a CSV file (`submission.csv`) for submission to Kaggle.

## Acknowledgements

This project uses the VGG-11 model from PyTorch's model zoo. Special thanks to the authors of the dataset and the PyTorch community for their contributions.

For more details, refer to the `code.ipynb` notebook.