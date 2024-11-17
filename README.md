# ML-Project-with-PyTorch-Breast-Cancer-Classification

Here's a detailed and professional README for your Breast Cancer Classification project:

---

# Breast Cancer Classification Using Scikit-Learn and PyTorch

This project is an exploration of machine learning techniques applied to classify breast cancer as malignant or benign. Leveraging both **scikit-learn** and **PyTorch**, the project demonstrates the full machine learning pipeline, from data preprocessing to model evaluation.

## Table of Contents
- [Introduction](#introduction)
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Implementation Details](#implementation-details)
- [Results](#results)
- [How to Use](#how-to-use)
- [Future Work](#future-work)
- [References](#references)

---

## Introduction
Breast cancer is one of the most common cancers globally, and early and accurate detection is crucial for effective treatment. This project aims to build a reliable classification model to distinguish between malignant and benign cases using advanced machine learning techniques.

---

## Project Overview
The project includes:
1. Data collection and preprocessing using **scikit-learn**.
2. Implementation of a fully connected neural network using **PyTorch**.
3. Evaluation of the model's performance on the test dataset.
4. Visualization of results and analysis of performance metrics.

---

## Dataset
The **Breast Cancer Wisconsin (Diagnostic) Dataset** from UCI Machine Learning Repository is used. Key details include:
- **Instances**: 569
- **Attributes**: 30 numerical features describing cell nuclei characteristics
- **Classes**: 
  - `0` - Malignant
  - `1` - Benign
- **Class Distribution**: 
  - Malignant: 212
  - Benign: 357

Features include measurements such as mean radius, texture, perimeter, area, and smoothness, among others. For a full description, refer to the dataset's [documentation](https://goo.gl/U2Uwz2).

---

## Methodology
1. **Data Preprocessing**:
   - Splitting the dataset into training and test sets.
   - Standardizing features to have a mean of 0 and standard deviation of 1 using `StandardScaler` from scikit-learn.
   - Converting data into PyTorch tensors for compatibility with the neural network.

2. **Model Architecture**:
   - A neural network with one hidden layer of 64 neurons.
   - Activation functions:
     - ReLU for non-linearity in the hidden layer.
     - Sigmoid for binary classification output.
   - Loss function: Binary Cross-Entropy Loss.
   - Optimizer: Adam optimizer for efficient gradient descent.

3. **Training**:
   - The model was trained for 100 epochs with a learning rate of 0.01.
   - Periodic evaluation of loss and accuracy during training.

4. **Evaluation**:
   - Assessing the model's performance on both training and test sets.
   - Calculating accuracy and visualizing results.

---

## Implementation Details
### Key Libraries
- **Scikit-learn**: For data preprocessing and splitting.
- **PyTorch**: For neural network construction and training.
- **Matplotlib**: For visualization.

### Neural Network Code Snippet
```python
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out
```

---

## Results
### Training Metrics:
- **Final Training Accuracy**: 99.78%
- **Final Loss**: 0.0107

### Test Metrics:
- **Accuracy on Test Data**: 99.34%

### Highlights:
- Excellent model performance with minimal overfitting.
- Effective handling of imbalanced class distribution.

---

## How to Use
1. **Prerequisites**:
   - Python 3.x
   - Libraries: scikit-learn, PyTorch, NumPy, Matplotlib, Pandas

2. **Steps**:
   - Clone the repository.
   - Install the required dependencies: `pip install -r requirements.txt`
   - Run the notebook `main.ipynb` to reproduce the results.

3. **Customization**:
   - Modify the neural network architecture in the `NeuralNet` class.
   - Experiment with different optimizers and learning rates.

---

## Future Work
- Extend the model to a multi-class classification problem for related datasets.
- Incorporate hyperparameter optimization techniques.
- Explore transfer learning for improved accuracy.

---

## References
- [Breast Cancer Wisconsin Dataset Documentation](https://goo.gl/U2Uwz2)
- PyTorch Documentation: [https://pytorch.org/docs/](https://pytorch.org/docs/)
- Scikit-learn Documentation: [https://scikit-learn.org/](https://scikit-learn.org/)

--- 
