# Customer Personality Analysis

This repository contains the code for **Customer Personality Analysis**, a project aimed at classifying customers based on their personalities using deep learning. The model is built using Keras and combines Convolutional Neural Networks (Conv1D), Bidirectional LSTM layers, and Dense layers to predict the personality class of a customer.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Training the Model](#training-the-model)
- [Evaluation](#evaluation)
- [Results](#results)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)

## Overview

Customer personality analysis is an important part of customer relationship management (CRM) as it helps businesses understand their customers better. In this project, we classify customers based on certain features, and the goal is to predict the personality category of each customer based on their data.

This project leverages deep learning techniques, specifically a hybrid model of **Conv1D** for feature extraction, **Bidirectional LSTM** for sequence modeling, and **Dense Layers** for classification.

## Dataset

The dataset used for this project contains various customer attributes, such as:

- Year_Birth
- Education
- Marital_Status
- Income etc.

The `Response` column is the target variable, which represents the customer’s personality category.

## Model Architecture

The model architecture consists of the following layers:

1. **Conv1D Layers**: Used for extracting local features from the data.
2. **MaxPooling1D**: For down-sampling the output of Conv1D layers.
3. **Bidirectional LSTM Layers**: Used to capture temporal dependencies from both directions.
4. **Batch Normalization**: To normalize the output and improve convergence.
5. **Dense Layers**: Fully connected layers for classification.
6. **Dropout Layers**: Added to prevent overfitting.

### Model Summary

```bash
Layer (type)                   Output Shape              Param #   
=================================================================
conv1d (Conv1D)                (None, 8, 64)             256   
max_pooling1d (MaxPooling1D)    (None, 4, 64)             0   
conv1d_1 (Conv1D)              (None, 2, 32)             6176  
max_pooling1d_1 (MaxPooling1D)  (None, 1, 32)             0   
bidirectional (Bidirectional)   (None, 256)               164864  
batch_normalization (BatchNorm) (None, 256)               1024  
dense (Dense)                  (None, 64)                16448   
dropout (Dropout)              (None, 64)                0   
dense_1 (Dense)                (None, 32)                2080  
dense_2 (Dense)                (None, 2)                 66  
=================================================================
Total params: 190,914
```

## Requirements

The following libraries are required to run the project:

- Python 3.x
- TensorFlow 2.x
- Keras
- Pandas
- Numpy
- Scikit-learn

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/SyabAhmad/customer-personality-analysis.git
   cd customer-personality-analysis
   ```
2. Install the required Python packages:

   ```bash
   pip install -r requirements.txt
   ```
3. (Optional) If you want to run the notebook in a virtual environment:

   ```bash
   python -m venv myenv
   source myenv/bin/activate  # For Linux/Mac
   myenv\Scripts\activate     # For Windows
   ```

## Usage

1. **Data Preprocessing**:
   Before training, the data must be preprocessed. You can load your dataset and prepare it for the model using the following steps:

   - Drop irrelevant columns
   - Scale or normalize features if necessary
   - One-hot encode categorical variables
   - Split the dataset into train, validation, and test sets
2. **Training**:
   Use the script `model.py` to train the model:

   ```bash
   python model.py
   ```
3. **Model Architecture**:
   The model architecture is defined using Keras. You can modify the architecture in the script depending on your specific needs.

## Training the Model

To train the model, the dataset is split into training, validation, and testing sets as shown below:

```python
from sklearn.model_selection import train_test_split

# Data Split
X = data.drop('Response', axis=1)
y = data['Response']

# One-hot encoding of labels
y = to_categorical(y)

# Split data into train, test, and validation sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
```

The model can be trained using the following command:

```python
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=64)
```

## Evaluation

Evaluate the trained model on the test data:

```python
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")
```

## Results

After training the model, you can expect the following metrics:

- **Training Accuracy**: ~97%
- **Validation Accuracy**: ~85%
- **Test Accuracy**: ~86%

These values may vary depending on hyperparameters, dataset size, and preprocessing techniques.

## Future Work

Potential improvements include:

- Fine-tuning hyperparameters like the number of filters in Conv1D, the number of LSTM units, and dropout rates.
- Implementing advanced architectures such as Transformer models for better accuracy.
- Conducting cross-validation for more robust performance evaluation.
- Collecting and using a larger dataset to improve generalization.

## Contributing

Feel free to fork the project and submit pull requests. Please make sure to follow the guidelines and ensure your contributions are well tested.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
