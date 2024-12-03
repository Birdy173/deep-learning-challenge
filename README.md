# Charity Outcome Optimization

This project focuses on building and optimizing a machine learning model to predict the success of charitable donations using TensorFlow and Keras. The notebooks analyze a dataset of charity application information and aim to improve prediction accuracy through preprocessing, feature engineering, and hyperparameter tuning.

## Repository Contents

### 1. `Starter_Code.ipynb`
This notebook contains the foundational steps for loading and preprocessing the dataset:
- Loading the dataset from a CSV file.
- Dropping non-beneficial columns such as `EIN` and `NAME`.
- Replacing infrequent categories in categorical columns (`APPLICATION_TYPE` and `CLASSIFICATION`) with "Other" to reduce dimensionality.
- Encoding categorical variables using one-hot encoding.
- Splitting the dataset into features (`X`) and target (`y`).
- Scaling the features using `StandardScaler` for better model performance.

### 2. `AlphabetSoupCharity_Optimization.ipynb`
This notebook extends the starter code by:
- Building a basic neural network model using TensorFlow's Sequential API.
- Performing hyperparameter optimization with Keras Tuner:
  - Tuning activation functions.
  - Tuning the number of neurons and layers.
- Compiling and training the model using binary cross-entropy loss and the Adam optimizer.
- Evaluating model performance on a test set.

## Installation

### Prerequisites
Ensure the following Python libraries are installed:
- `tensorflow`
- `keras-tuner`
- `scikit-learn`
- `pandas`
- `numpy`

You can install these dependencies using:
```bash
pip install tensorflow keras-tuner scikit-learn pandas numpy
```

### Data
The dataset is fetched from a remote source during runtime:
`https://static.bc-edx.com/data/dl-1-2/m21/lms/starter/charity_data.csv`

### Running the Project
1. Open the Jupyter notebooks in your Python environment.
2. Execute the cells sequentially to preprocess the data, build models, and evaluate performance.

## Usage
This repository is designed for:
- Data preprocessing workflows for machine learning.
- Understanding hyperparameter tuning with Keras Tuner.
- Developing neural network models for classification tasks.
