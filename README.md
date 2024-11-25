# TextClassification-ML

# Tweet Sentiment Analysis

This repository contains a machine learning model for classifying the sentiment of tweets as **Positive**, **Neutral**, or **Negative**. The model uses advanced Natural Language Processing (NLP) techniques to analyze and categorize tweet content.

## Key Features

- **Data Processing**: Clean and preprocess tweet text by removing noise, stopwords, and applying tokenization and stemming.
- **Text Vectorization**: Converts tweet text into numerical features using **TF-IDF** (Term Frequency-Inverse Document Frequency).
- **Modeling**: Evaluates several machine learning models including:
  - Logistic Regression
  - K-Nearest Neighbors (KNN)
  - Random Forest
  - Naive Bayes
  - Artificial Neural Network (ANN)
  
- **Evaluation**: Model performance is measured using accuracy, classification reports, and confusion matrices.

## Model Performance

| **Model**                  | **Accuracy** |
|----------------------------|--------------|
| Logistic Regression         | 70.99%       |
| K-Nearest Neighbors (KNN)   | 48.50%       |
| Random Forest Classifier    | 70.83%       |
| Multinomial Naive Bayes     | 64.12%       |
| **Neural Network**          | **98.36%**   |

## Requirements

- Python 3.x
- Libraries: `numpy`, `pandas`, `scikit-learn`, `tensorflow`, `nltk`, `matplotlib`
