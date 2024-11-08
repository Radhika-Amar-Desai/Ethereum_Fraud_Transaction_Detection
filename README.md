# Ethereum Fraud Transaction Detection

This project is focused on detecting fraudulent transactions on the Ethereum blockchain. It utilizes a machine learning model trained on historical transaction data to classify transactions as fraudulent or legitimate. This repository contains code for preprocessing Ethereum transaction data, training a model, and evaluating the performance of the model on a test dataset.

## Features

- **Data Preprocessing:** Cleans and preprocesses raw Ethereum transaction data.
- **Feature Engineering:** Extracts relevant features from transaction records.
- **Model Training:** Trains a machine learning model to identify fraudulent transactions.
- **Evaluation:** Tests the model on new data to assess accuracy and reliability.

## Installation

### Prerequisites

- Python 3.8 or higher
- [Pandas](https://pandas.pydata.org/) for data manipulation
- [NumPy](https://numpy.org/) for numerical operations
- [scikit-learn](https://scikit-learn.org/) for machine learning model training and evaluation
- [Matplotlib](https://matplotlib.org/) and [Seaborn](https://seaborn.pydata.org/) for data visualization (optional)

### Setup

Clone this repository to your local machine and navigate into the project directory:

```bash
git clone https://github.com/yourusername/ethereum-fraud-detection.git
cd ethereum-fraud-detection
```

## Usage

### 1. Data Collection

You can download the dataset from Kaggle (https://www.kaggle.com/datasets/vagifa/ethereum-frauddetection-dataset)
Obtain the Ethereum transaction data (CSV format) and place it in the `DataAnalysis/` directory.

### 2. Train the Model

You can run the `CompleteFramework.ipynb` notebook to see the results of the complete methodology used or run the `Model.ipynb` notebook to see the results of the model without hyperparameter finetuning.
