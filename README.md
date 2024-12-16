# Tweet Sentiment Analysis with Machine Learning and Deep Learning

## Overview

The goal of this project is to build a machine learning model to accurately predict tweet sentiment. The project utilizes various machine learning models and deep learning techniques, with an emphasis on preprocessing, feature extraction, and model evaluation.

## Results

The following visualizations and tables provide insights into the model's performance:

### Model Performance

| **Model**                  | **Accuracy** |
|----------------------------|--------------|
| Logistic Regression         | 70.99%       |
| K-Nearest Neighbors (KNN)   | 48.50%       |
| Random Forest Classifier    | 70.83%       |
| Multinomial Naive Bayes     | 64.12%       |
| **Neural Network**          | **98.36%**   |

## Features

- **Data Processing**: Clean and preprocess tweet text by removing noise, stopwords, and applying tokenization and stemming.
- **Text Vectorization**: Converts tweet text into numerical features using **TF-IDF** (Term Frequency-Inverse Document Frequency).
- **Modeling**: Evaluates several machine learning models including:
  - Logistic Regression
  - K-Nearest Neighbors (KNN)
  - Random Forest
  - Naive Bayes
  - Artificial Neural Network (ANN)
  
- **Evaluation**: Model performance is measured using accuracy, classification reports, and confusion matrices.

## Sprint Features

### Sprint 1: Data Collection and Preprocessing
- Collect and preprocess the tweet dataset.
- Clean the data by removing unnecessary symbols, stopwords, and applying text normalization techniques like stemming and lemmatization.
- **Deliverable**: Cleaned dataset ready for training.

### Sprint 2: Feature Engineering
- Convert text data into numerical features using TF-IDF vectorization.
- Explore and experiment with different feature extraction techniques to improve model accuracy.
- **Deliverable**: Feature matrix ready for model training.

### Sprint 3: Model Training and Evaluation
- Train several machine learning models (Logistic Regression, KNN, Random Forest, Naive Bayes) and evaluate their performance using accuracy, precision, recall, and F1 score.
- Compare the performance of different models to determine the best one for the task.
- **Deliverable**: Best-performing model selected and evaluated.

### Sprint 4: Deep Learning Model Implementation
- Implement an artificial neural network (ANN) for tweet sentiment classification.
- Tune hyperparameters and evaluate the ANN's performance.
- **Deliverable**: Trained neural network model with high accuracy.

## Conclusion

This project demonstrates how machine learning and deep learning models can be applied to sentiment analysis tasks. The neural network model performed the best, achieving an accuracy of 98.36%. Future improvements could involve exploring more advanced deep learning models or using more complex feature extraction methods to further enhance model performance.
