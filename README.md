# Email Spam Classification using Logistic Regression in R

## Overview
This project focuses on classifying emails as **Spam** or **Non-Spam** using **Logistic Regression** and **Regularized Ridge Logistic Regression** in R. The system applies machine learning techniques and statistical modeling to detect spam emails with high accuracy and strong predictive performance.

The project demonstrates a complete machine learning pipeline including:

- Data preprocessing
- Feature scaling
- Correlation analysis
- Logistic Regression modeling
- Ridge Regularization
- Cross-validation
- ROC-AUC analysis
- Model evaluation
- Feature importance analysis

The project uses the built-in **spam dataset** from the `kernlab` package and evaluates both standard and regularized logistic regression models for email spam detection.

---

# Problem Statement

Spam emails are one of the major challenges in modern communication systems. Traditional rule-based spam filters often fail to adapt to evolving spam patterns.

This project builds a machine learning-based spam detection system capable of:

- Automatically classifying emails
- Reducing false positives
- Improving email security
- Detecting spam patterns efficiently

The classification model analyzes multiple textual and statistical features extracted from emails and predicts whether an email is:

- Spam
- Non-Spam

The project demonstrates how machine learning can be applied in real-world cybersecurity and email filtering systems.

---

# Dataset Information

The project uses the **Spam Dataset** available in the `kernlab` package in R.

### Dataset Characteristics

- Contains multiple numeric features extracted from emails
- Binary target classification:
  - `spam`
  - `nonspam`
- Used widely for spam classification research

---

# Technologies Used

## Programming Language

- R

## Libraries & Packages

- tidyverse
- caret
- tm
- SnowballC
- e1071
- ROCR
- pROC
- kernlab
- glmnet

---

# Concepts Used

- Logistic Regression
- Ridge Regularization
- Binary Classification
- ROC Curve
- AUC Analysis
- Cross Validation
- Feature Scaling
- Confusion Matrix
- Feature Importance
- Predictive Analytics

---

# Machine Learning Workflow

## 1. Data Loading & Exploration

The spam dataset was loaded from the `kernlab` package and explored for:

- Dataset dimensions
- Missing values
- Class distribution
- Feature analysis

```r
data(spam)
```

---

## 2. Data Preprocessing

The project performs several preprocessing operations:

### Target Variable Cleaning

To avoid naming conflicts in R:

```r
spam$spam_label <- ifelse(spam$type == "spam", "spam", "nonspam")
```

### Additional Preprocessing Steps

- Removed original target column
- Checked near-zero variance predictors
- Removed highly correlated features
- Standardized numerical features

---

## 3. Feature Scaling

Feature scaling was performed using:

```r
preProcess(method = c("center", "scale"))
```

This ensures all features contribute equally during model training.

---

## 4. Train-Test Split

Dataset was split into:

- 70% Training Data
- 30% Testing Data

Using stratified sampling:

```r
createDataPartition()
```

---

## 5. Standard Logistic Regression Model

A standard logistic regression model was built using:

```r
glm(
  spam_label ~ .,
  data = train_data,
  family = binomial(link = "logit")
)
```

The model predicts probabilities for spam classification.

---

## 6. Regularized Ridge Logistic Regression

To reduce overfitting and improve generalization, Ridge Regression was implemented using:

```r
method = "glmnet"
alpha = 0
```

The project uses:

- 10-Fold Cross Validation
- Lambda tuning
- ROC optimization

---

# Logistic Regression

Logistic Regression is a supervised machine learning algorithm used for binary classification problems.

The sigmoid function is:

```math
P(Y=1)=\frac{1}{1+e^{-(\beta_0+\beta_1x_1+\beta_2x_2+\cdots+\beta_nx_n)}}
```

The algorithm predicts the probability that an email belongs to the spam category.

---

# Model Evaluation Metrics

The project evaluates models using:

- Accuracy
- Precision
- Recall
- Specificity
- F1 Score
- ROC-AUC

Evaluation performed using:

```r
confusionMatrix()
```

and ROC curve analysis.

---

# ROC Curve & AUC Analysis

ROC curves were generated for:

- Standard Logistic Regression
- Ridge Logistic Regression

AUC scores were compared to determine the better-performing model.

```r
roc()
auc()
```

---

# Feature Importance Analysis

Feature importance was analyzed using:

- Logistic regression coefficients
- caret `varImp()` function

Top features influencing spam prediction were identified automatically.

---

# Model Saving

Trained models were saved using:

```r
saveRDS()
```

Saved files include:

- `logistic_spam_model.rds`
- `ridge_spam_model.rds`
- `preprocessing_params.rds`

---

# Key Highlights

- Implemented complete machine learning workflow in R
- Built both standard and regularized logistic regression models
- Applied feature scaling and preprocessing techniques
- Performed ROC-AUC analysis
- Implemented cross-validation for hyperparameter tuning
- Generated feature importance rankings
- Built reusable preprocessing pipeline
- Demonstrated real-world spam filtering use case

---

# Project Structure

```bash
email-spam-classification-r/
│
├── email_spam_classification.R
├── logistic_spam_model.rds
├── ridge_spam_model.rds
├── preprocessing_params.rds
├── roc_curves.png
├── README.md
└── outputs/
```

---

# Installation & Setup

## Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/email-spam-classification-r.git
```

---

## Install Required Packages

```r
install.packages(c(
  "tidyverse",
  "caret",
  "tm",
  "SnowballC",
  "e1071",
  "ROCR",
  "pROC",
  "kernlab",
  "glmnet"
))
```

---

## Run the Project

```r
source("email_spam_classification.R")
```

---

# Sample Output

```bash
EMAIL SPAM CLASSIFICATION USING LOGISTIC REGRESSION

Training Standard Logistic Regression Model...
✓ Model training completed!

Training Regularized Logistic Regression...
✓ Regularized model training completed!

AUC for Standard Logistic Model: 0.98
AUC for Regularized Ridge Model: 0.99

Recommendation:
Use the Ridge model for production deployment
```

---

# Learning Outcomes

Through this project, I gained practical experience in:

- Logistic Regression
- Ridge Regularization
- Machine Learning Pipelines
- Feature Engineering
- ROC-AUC Analysis
- Model Evaluation
- Predictive Analytics
- R Programming
- Spam Detection Systems

---

# Limitations

- Logistic Regression assumes linear decision boundaries
- Model performance depends on feature engineering quality
- Spam patterns may evolve over time
- High-dimensional data can increase training complexity

---

# Future Improvements

- Implement NLP-based text preprocessing
- Add TF-IDF vectorization
- Compare with:
  - Naive Bayes
  - Random Forest
  - XGBoost
  - SVM
- Build real-time spam filtering API
- Deploy using Shiny Dashboard
- Add email visualization analytics

---

# Author

**Sukumar Erugadindla**  
B.Tech – Computer Science Engineering  
Machine Learning Developer | Data Science Enthusiast | Full Stack Developer

GitHub: https://github.com/Sukumar5705

---

# References

- R Documentation
- caret Package Documentation
- glmnet Documentation
- kernlab Package
- Logistic Regression Research Papers

---

# License

This project is open-source and available for educational and research purposes.
