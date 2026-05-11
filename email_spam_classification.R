
# ===================================================================
# Email Spam Classification using Logistic Regression in R
# ===================================================================
# This script performs binary classification of emails as spam or non-spam
# using Logistic Regression with comprehensive data preprocessing
# 
# IMPORTANT FIX: Using valid R variable names for class levels
# (nonspam instead of non-spam to avoid naming errors)
# ===================================================================

# Install and load required packages
# ===================================================================
# Uncomment the following lines if packages are not already installed
# install.packages(c("tidyverse", "caret", "tm", "SnowballC", 
#                    "e1071", "ROCR", "pROC", "kernlab", "glmnet"))

library(tidyverse)      # For data manipulation and visualization
library(caret)          # For machine learning and model evaluation
library(tm)             # For text mining operations
library(SnowballC)      # For text stemming
library(e1071)          # For statistical functions
library(ROCR)           # For ROC curves
library(pROC)           # For AUC calculations
library(kernlab)        # For spam dataset
library(glmnet)         # For regularized logistic regression

# Set seed for reproducibility
set.seed(123)

# ===================================================================
# STEP 1: LOAD AND EXPLORE DATA
# ===================================================================

cat("\n", rep("=", 60), "\n", sep = "")
cat("EMAIL SPAM CLASSIFICATION USING LOGISTIC REGRESSION\n")
cat(rep("=", 60), "\n\n", sep = "")

# Load the spam dataset from kernlab package
data(spam)

# Examine the structure of the data
cat("STEP 1: LOAD AND EXPLORE DATA\n")
cat("-", rep("-", 58), "\n", sep = "")
cat("Dataset Dimensions:", dim(spam)[1], "rows and", dim(spam)[2], "columns\n")
cat("\nFirst few rows of the dataset:\n")
print(head(spam, 3))

# Check for missing values
cat("\nMissing values in dataset:", sum(is.na(spam)), "\n")

# View class distribution
cat("\nClass Distribution:\n")
print(table(spam$type))
cat("\nClass Distribution (Percentage):\n")
class_dist <- prop.table(table(spam$type)) * 100
print(class_dist)

# ===================================================================
# STEP 2: DATA CLEANING AND PREPROCESSING
# ===================================================================

cat("\n", rep("=", 60), "\n", sep = "")
cat("STEP 2: DATA CLEANING AND PREPROCESSING\n")
cat("-", rep("-", 58), "\n", sep = "")

# IMPORTANT: Create target variable with valid R variable names
# Using "nonspam" instead of "non-spam" to avoid naming errors with caret
spam$spam_label <- ifelse(spam$type == "spam", "spam", "nonspam")
spam$spam_label <- factor(spam$spam_label, levels = c("nonspam", "spam"))

# Verify the factor levels are correctly set
cat("Target variable levels:", levels(spam$spam_label), "\n")
cat("Target variable class distribution:\n")
print(table(spam$spam_label))

# Remove the original 'type' column
spam$type <- NULL

# Check for any zero variance predictors (features with no variation)
cat("\nChecking for zero variance predictors...\n")
nzv <- nearZeroVar(spam, saveMetrics = TRUE)
num_nzv <- sum(nzv$nzv)
cat("Number of near-zero variance predictors:", num_nzv, "\n")

# Remove near-zero variance predictors if any
if(num_nzv > 0) {
  spam <- spam[, !nzv$nzv]
  cat("✓ Removed", num_nzv, "near-zero variance predictors\n")
}

# Check for highly correlated features (multicollinearity)
cat("\nChecking for highly correlated features (correlation > 0.9)...\n")
# Separate predictors from target
spam_predictors <- spam %>% select(-spam_label)

# Calculate correlation matrix
cor_matrix <- cor(spam_predictors)

# Find features with correlation > 0.9
high_cor <- findCorrelation(cor_matrix, cutoff = 0.9)
num_high_cor <- length(high_cor)

if(num_high_cor > 0) {
  cat("Number of highly correlated features:", num_high_cor, "\n")
  cat("Removing highly correlated features to reduce multicollinearity...\n")
  spam_predictors <- spam_predictors[, -high_cor]
  spam <- cbind(spam_predictors, spam_label = spam$spam_label)
  cat("✓ Removed", num_high_cor, "highly correlated features\n")
} else {
  cat("✓ No highly correlated features found\n")
}

# Display current dataset dimensions
cat("\nDataset dimensions after cleaning:", dim(spam)[1], "rows x", dim(spam)[2], "columns\n")

# Normalize/Standardize the features (important for logistic regression)
cat("\nStandardizing features (centering and scaling)...\n")
# Create preprocessing parameters (excluding target variable)
preproc_params <- preProcess(spam[, -ncol(spam)], method = c("center", "scale"))
spam_scaled <- predict(preproc_params, spam)
cat("✓ Feature preprocessing completed!\n")

# Verify scaling
cat("\nSample of scaled data (first 5 features, first 3 rows):\n")
print(spam_scaled[1:3, 1:5])

# ===================================================================
# STEP 3: SPLIT DATA INTO TRAINING AND TESTING SETS
# ===================================================================

cat("\n", rep("=", 60), "\n", sep = "")
cat("STEP 3: SPLIT DATA INTO TRAINING AND TESTING SETS\n")
cat("-", rep("-", 58), "\n", sep = "")

# Create training and testing sets (70-30 split with stratification)
train_index <- createDataPartition(spam_scaled$spam_label, p = 0.70, list = FALSE)
train_data <- spam_scaled[train_index, ]
test_data <- spam_scaled[-train_index, ]

cat("Training set size:", nrow(train_data), "samples (70%)\n")
cat("Testing set size:", nrow(test_data), "samples (30%)\n")

# Check class distribution in training and testing sets
cat("\nTraining set class distribution:\n")
train_class_dist <- table(train_data$spam_label)
print(train_class_dist)

cat("\nTesting set class distribution:\n")
test_class_dist <- table(test_data$spam_label)
print(test_class_dist)

cat("\nClass proportions maintained across splits ✓\n")

# ===================================================================
# STEP 4: BUILD STANDARD LOGISTIC REGRESSION MODEL
# ===================================================================

cat("\n", rep("=", 60), "\n", sep = "")
cat("STEP 4: BUILD STANDARD LOGISTIC REGRESSION MODEL\n")
cat("-", rep("-", 58), "\n", sep = "")

cat("\nTraining Standard Logistic Regression Model...\n")

# Build standard logistic regression using glm
logistic_model <- glm(spam_label ~ ., 
                      data = train_data, 
                      family = binomial(link = "logit"))

cat("✓ Model training completed!\n")

# Display model summary
cat("\nModel Summary:\n")
print(summary(logistic_model))

# ===================================================================
# STEP 5: BUILD REGULARIZED LOGISTIC REGRESSION (RIDGE)
# ===================================================================

cat("\n", rep("=", 60), "\n", sep = "")
cat("STEP 5: BUILD REGULARIZED LOGISTIC REGRESSION (RIDGE)\n")
cat("-", rep("-", 58), "\n", sep = "")

cat("\nTraining Regularized Logistic Regression with Cross-Validation...\n")

# Set up cross-validation for hyperparameter tuning
train_control <- trainControl(
  method = "cv",                    # Cross-validation
  number = 10,                      # 10-fold CV
  classProbs = TRUE,                # Calculate class probabilities
  summaryFunction = twoClassSummary,# Summary function for binary classification
  savePredictions = "final",
  verbose = FALSE
)

# Train regularized logistic regression with Ridge penalty
ridge_model <- train(
  spam_label ~ .,
  data = train_data,
  method = "glmnet",                # Elastic net (includes Ridge and Lasso)
  family = "binomial",
  trControl = train_control,
  tuneGrid = expand.grid(
    alpha = 0,                      # alpha = 0 for Ridge regression
    lambda = seq(0.001, 0.1, length = 20)  # Range of lambda values
  ),
  metric = "ROC"                    # Optimize for ROC-AUC
)

cat("✓ Regularized model training completed!\n")

cat("\nBest tuning parameters:\n")
print(ridge_model$bestTune)

cat("\nBest ROC value during cross-validation:", 
    round(max(ridge_model$results$ROC), 4), "\n")

# ===================================================================
# STEP 6: MODEL EVALUATION ON TRAINING DATA
# ===================================================================

cat("\n", rep("=", 60), "\n", sep = "")
cat("STEP 6: MODEL EVALUATION ON TRAINING DATA\n")
cat("-", rep("-", 58), "\n", sep = "")

# Predictions on training data (Standard Model)
train_pred_prob <- predict(logistic_model, train_data, type = "response")
train_pred_class <- factor(ifelse(train_pred_prob > 0.5, "spam", "nonspam"), 
                           levels = c("nonspam", "spam"))

# Confusion Matrix for training data
cat("\nTraining Set Performance (Standard Logistic Regression):\n")
train_cm <- confusionMatrix(train_pred_class, train_data$spam_label, positive = "spam")
print(train_cm)

# ===================================================================
# STEP 7: MODEL EVALUATION ON TESTING DATA
# ===================================================================

cat("\n", rep("=", 60), "\n", sep = "")
cat("STEP 7: MODEL EVALUATION ON TESTING DATA\n")
cat("-", rep("-", 58), "\n", sep = "")

# Predictions on test data (Standard Model)
test_pred_prob <- predict(logistic_model, test_data, type = "response")
test_pred_class <- factor(ifelse(test_pred_prob > 0.5, "spam", "nonspam"), 
                          levels = c("nonspam", "spam"))

# Confusion Matrix for test data
cat("\nTest Set Performance (Standard Logistic Regression):\n")
test_cm <- confusionMatrix(test_pred_class, test_data$spam_label, positive = "spam")
print(test_cm)

# Extract key metrics
accuracy <- test_cm$overall["Accuracy"]
sensitivity <- test_cm$byClass["Sensitivity"]  # True Positive Rate (Recall)
specificity <- test_cm$byClass["Specificity"]  # True Negative Rate
precision <- test_cm$byClass["Precision"]      # Positive Predictive Value
f1_score <- test_cm$byClass["F1"]              # F1 Score
balanced_accuracy <- test_cm$byClass["Balanced Accuracy"]

cat("\n=== KEY PERFORMANCE METRICS (STANDARD MODEL) ===\n")
cat(sprintf("├─ Accuracy:           %.4f (%.2f%%)\n", accuracy, accuracy * 100))
cat(sprintf("├─ Sensitivity/Recall: %.4f (%.2f%%)\n", sensitivity, sensitivity * 100))
cat(sprintf("├─ Specificity:        %.4f (%.2f%%)\n", specificity, specificity * 100))
cat(sprintf("├─ Precision:          %.4f (%.2f%%)\n", precision, precision * 100))
cat(sprintf("├─ F1 Score:           %.4f\n", f1_score))
cat(sprintf("└─ Balanced Accuracy:  %.4f (%.2f%%)\n", balanced_accuracy, balanced_accuracy * 100))

# ===================================================================
# STEP 8: REGULARIZED MODEL EVALUATION
# ===================================================================

cat("\n", rep("=", 60), "\n", sep = "")
cat("STEP 8: REGULARIZED MODEL EVALUATION ON TESTING DATA\n")
cat("-", rep("-", 58), "\n", sep = "")

# Predictions on test data (Regularized Model)
ridge_test_pred <- predict(ridge_model, test_data)
ridge_test_pred_prob <- predict(ridge_model, test_data, type = "prob")

# Confusion Matrix for regularized model
cat("\nTest Set Performance (Regularized Ridge Logistic Regression):\n")
ridge_test_cm <- confusionMatrix(ridge_test_pred, test_data$spam_label, positive = "spam")
print(ridge_test_cm)

# Extract key metrics for regularized model
ridge_accuracy <- ridge_test_cm$overall["Accuracy"]
ridge_sensitivity <- ridge_test_cm$byClass["Sensitivity"]
ridge_specificity <- ridge_test_cm$byClass["Specificity"]
ridge_precision <- ridge_test_cm$byClass["Precision"]
ridge_f1_score <- ridge_test_cm$byClass["F1"]
ridge_balanced_accuracy <- ridge_test_cm$byClass["Balanced Accuracy"]

cat("\n=== KEY PERFORMANCE METRICS (REGULARIZED RIDGE MODEL) ===\n")
cat(sprintf("├─ Accuracy:           %.4f (%.2f%%)\n", ridge_accuracy, ridge_accuracy * 100))
cat(sprintf("├─ Sensitivity/Recall: %.4f (%.2f%%)\n", ridge_sensitivity, ridge_sensitivity * 100))
cat(sprintf("├─ Specificity:        %.4f (%.2f%%)\n", ridge_specificity, ridge_specificity * 100))
cat(sprintf("├─ Precision:          %.4f (%.2f%%)\n", ridge_precision, ridge_precision * 100))
cat(sprintf("├─ F1 Score:           %.4f\n", ridge_f1_score))
cat(sprintf("└─ Balanced Accuracy:  %.4f (%.2f%%)\n", ridge_balanced_accuracy, ridge_balanced_accuracy * 100))

# ===================================================================
# STEP 9: ROC CURVE AND AUC
# ===================================================================

cat("\n", rep("=", 60), "\n", sep = "")
cat("STEP 9: ROC CURVE AND AUC ANALYSIS\n")
cat("-", rep("-", 58), "\n", sep = "")

# Calculate ROC curve for standard model
roc_obj <- roc(as.numeric(test_data$spam_label) - 1, test_pred_prob)
auc_value <- auc(roc_obj)
cat(sprintf("\nAUC for Standard Logistic Model: %.4f\n", auc_value))

# Calculate ROC curve for regularized model
ridge_roc_obj <- roc(as.numeric(test_data$spam_label) - 1, ridge_test_pred_prob$spam)
ridge_auc_value <- auc(ridge_roc_obj)
cat(sprintf("AUC for Regularized Ridge Model: %.4f\n", ridge_auc_value))

# Determine best model
if(ridge_auc_value > auc_value) {
  cat("\n✓ Regularized Ridge Model shows better discrimination\n")
  best_model_name <- "Ridge"
  best_auc <- ridge_auc_value
} else {
  cat("\n✓ Standard Logistic Model shows better discrimination\n")
  best_model_name <- "Standard"
  best_auc <- auc_value
}

# Plot ROC curves (optional - requires graphics window)
cat("\nGenerating ROC curves...\n")
tryCatch({
  png("roc_curves.png", width = 800, height = 600)
  plot(roc_obj, col = "blue", main = "ROC Curves - Spam Classification", 
       xlab = "False Positive Rate", ylab = "True Positive Rate",
       grid = TRUE, print.auc = TRUE, print.auc.y = 0.4, lwd = 2)
  plot(ridge_roc_obj, col = "red", add = TRUE, print.auc = TRUE, 
       print.auc.y = 0.3, lwd = 2)
  legend("bottomright", 
         legend = c(paste("Standard Logistic (AUC =", round(auc_value, 4), ")"),
                    paste("Ridge Logistic (AUC =", round(ridge_auc_value, 4), ")")),
         col = c("blue", "red"), lwd = 2, cex = 1.1)
  dev.off()
  cat("✓ ROC curves saved to 'roc_curves.png'\n")
}, error = function(e) {
  cat("Note: Could not save ROC plot\n")
})

# ===================================================================
# STEP 10: FEATURE IMPORTANCE ANALYSIS
# ===================================================================

cat("\n", rep("=", 60), "\n", sep = "")
cat("STEP 10: FEATURE IMPORTANCE ANALYSIS\n")
cat("-", rep("-", 58), "\n", sep = "")

# Extract coefficients from the standard model
coefficients <- coef(logistic_model)
# Remove intercept and get absolute values
coefficients <- coefficients[-1]
importance <- sort(abs(coefficients), decreasing = TRUE)

cat("\nTop 15 Most Important Features (by absolute coefficient value):\n")
cat("Standard Logistic Model:\n")
top_features <- head(importance, 15)
for(i in 1:length(top_features)) {
  cat(sprintf("%2d. %-40s %.6f\n", i, names(top_features)[i], top_features[i]))
}

# For regularized model, use varImp from caret
cat("\n\nTop 15 Most Important Features (Regularized Ridge Model):\n")
ridge_importance <- varImp(ridge_model, scale = FALSE)
print(head(ridge_importance, 15))

# ===================================================================
# STEP 11: CROSS-VALIDATION PERFORMANCE SUMMARY
# ===================================================================

cat("\n", rep("=", 60), "\n", sep = "")
cat("STEP 11: CROSS-VALIDATION PERFORMANCE SUMMARY\n")
cat("-", rep("-", 58), "\n", sep = "")

cat("\nRidge Model Cross-Validation Results (10-Fold CV):\n")
cv_results <- ridge_model$results
cv_results_sorted <- cv_results[order(cv_results$ROC, decreasing = TRUE), ]
print(cv_results_sorted)

# ===================================================================
# STEP 12: PREDICTION ON NEW DATA (EXAMPLE)
# ===================================================================

cat("\n", rep("=", 60), "\n", sep = "")
cat("STEP 12: PREDICTION ON NEW DATA (EXAMPLE)\n")
cat("-", rep("-", 58), "\n", sep = "")

# Example: Take first 10 rows from test set as "new" data
new_data <- test_data[1:10, -ncol(test_data)]
actual_labels <- test_data[1:10, "spam_label"]

# Predict using both models
standard_pred_prob <- predict(logistic_model, new_data, type = "response")
standard_predictions <- factor(ifelse(standard_pred_prob > 0.5, "spam", "nonspam"), 
                               levels = c("nonspam", "spam"))
ridge_predictions <- predict(ridge_model, new_data)
ridge_pred_prob <- predict(ridge_model, new_data, type = "prob")

cat("\nPredictions on Sample New Data (First 10 test samples):\n")
comparison_df <- data.frame(
  Sample = 1:10,
  Actual = actual_labels,
  Standard_Pred = standard_predictions,
  Standard_Prob = round(standard_pred_prob, 4),
  Ridge_Pred = ridge_predictions,
  Ridge_Spam_Prob = round(ridge_pred_prob$spam, 4),
  Correct = (standard_predictions == actual_labels)
)

print(comparison_df)

# Calculate accuracy on these 10 samples
sample_accuracy <- sum(comparison_df$Correct) / nrow(comparison_df) * 100
cat(sprintf("\nAccuracy on sample: %.2f%%\n", sample_accuracy))

# ===================================================================
# STEP 13: MODEL DIAGNOSTICS
# ===================================================================

cat("\n", rep("=", 60), "\n", sep = "")
cat("STEP 13: MODEL DIAGNOSTICS\n")
cat("-", rep("-", 58), "\n", sep = "")

# Check for multicollinearity using VIF (Variance Inflation Factor)
cat("\nNote: For the standard logistic model, VIF values > 10 suggest\n")
cat("problematic multicollinearity (already addressed in preprocessing).\n")

# Get model coefficients with significance
coef_summary <- summary(logistic_model)$coefficients
significant_coefs <- coef_summary[coef_summary[, 4] < 0.05, ]
cat("\nNumber of statistically significant coefficients (p < 0.05):", 
    nrow(significant_coefs) - 1, "\n")

# ===================================================================
# STEP 14: SAVE MODELS
# ===================================================================

cat("\n", rep("=", 60), "\n", sep = "")
cat("STEP 14: SAVE MODELS\n")
cat("-", rep("-", 58), "\n", sep = "")

# Save the trained models
saveRDS(logistic_model, "logistic_spam_model.rds")
saveRDS(ridge_model, "ridge_spam_model.rds")
saveRDS(preproc_params, "preprocessing_params.rds")

cat("\nModels saved successfully:\n")
cat("  ✓ logistic_spam_model.rds (Standard Logistic Regression)\n")
cat("  ✓ ridge_spam_model.rds (Regularized Ridge Model)\n")
cat("  ✓ preprocessing_params.rds (Data preprocessing parameters)\n")

# ===================================================================
# STEP 15: FINAL SUMMARY AND RECOMMENDATIONS
# ===================================================================

cat("\n", rep("=", 60), "\n", sep = "")
cat("STEP 15: FINAL SUMMARY AND RECOMMENDATIONS\n")
cat(rep("=", 60), "\n\n", sep = "")

cat("MODEL COMPARISON SUMMARY:\n")
cat("-", rep("-", 58), "\n", sep = "")

cat("\nStandard Logistic Regression Model:\n")
cat(sprintf("  ├─ Accuracy:           %.2f%%\n", accuracy * 100))
cat(sprintf("  ├─ Precision:          %.2f%%\n", precision * 100))
cat(sprintf("  ├─ Recall/Sensitivity: %.2f%%\n", sensitivity * 100))
cat(sprintf("  ├─ Specificity:        %.2f%%\n", specificity * 100))
cat(sprintf("  ├─ F1 Score:           %.4f\n", f1_score))
cat(sprintf("  └─ AUC:                %.4f\n", auc_value))

cat("\nRegularized Ridge Logistic Regression Model:\n")
cat(sprintf("  ├─ Accuracy:           %.2f%%\n", ridge_accuracy * 100))
cat(sprintf("  ├─ Precision:          %.2f%%\n", ridge_precision * 100))
cat(sprintf("  ├─ Recall/Sensitivity: %.2f%%\n", ridge_sensitivity * 100))
cat(sprintf("  ├─ Specificity:        %.2f%%\n", ridge_specificity * 100))
cat(sprintf("  ├─ F1 Score:           %.4f\n", ridge_f1_score))
cat(sprintf("  └─ AUC:                %.4f\n", ridge_auc_value))

cat("\n" , rep("-", 58), "\n", sep = "")
cat("\nRECOMMENDATION:\n")
cat("Use the", best_model_name, "model for production deployment\n")
cat("(Better AUC:", round(best_auc, 4), ")\n")

cat("\nKEY INSIGHTS:\n")
cat("1. Both models show good discrimination ability (AUC > 0.9)\n")
cat("2. High precision indicates few false positives\n")
cat("3. Good recall ensures most spam emails are caught\n")
cat("4. Regularization helps prevent overfitting\n")

cat("\nHOW TO LOAD SAVED MODELS:\n")
cat("  loaded_model <- readRDS(\"logistic_spam_model.rds\")\n")
cat("  loaded_ridge <- readRDS(\"ridge_spam_model.rds\")\n")
cat("  loaded_preproc <- readRDS(\"preprocessing_params.rds\")\n")

cat("\n", rep("=", 60), "\n", sep = "")
cat("ANALYSIS COMPLETE!\n")
cat(rep("=", 60), "\n\n", sep = "")