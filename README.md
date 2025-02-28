# Diabetes Prediction with Machine Learning

This repository contains Python scripts for predicting diabetes using the 2015 Behavioral Risk Factor Surveillance System (BRFSS) dataset. The scripts preprocess the data with Principal Component Analysis (PCA) and implement various classification techniques, including Logistic Regression, Naive Bayes, Support Vector Machines (SVM), and ensemble voting (hard and soft). The goal is to classify individuals as diabetic or non-diabetic based on health indicators.

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
  - [Script 1: Ensemble Voting with Logistic Regression and Naive Bayes](#script-1-ensemble-voting-with-logistic-regression-and-naive-bayes)
  - [Script 2: Gaussian Naive Bayes](#script-2-gaussian-naive-bayes)
  - [Script 3: Support Vector Machine (SVM)](#script-3-support-vector-machine-svm)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Data Preprocessing:** Standardizes data and applies PCA for dimensionality reduction.
- **Classification Models:**
  - Logistic Regression with gradient descent.
  - Gaussian Naive Bayes with class weighting.
  - Support Vector Machine (SVM) using Sequential Minimal Optimization (SMO).
  - Ensemble voting (hard and soft) combining multiple models.
- **Evaluation Metrics:** Accuracy, precision, recall, and F1-score.
- **Balanced Dataset:** Handles class imbalance by sampling equal numbers of diabetic and non-diabetic cases.

## Requirements

- Python 3.8+
- Libraries:
  - `numpy`
  - `collections` (for `Counter` in ensemble voting)

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
