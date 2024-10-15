# -*- coding: utf-8 -*-

"""
__author__ = "Ana B. Caniceiro, Ana M. B. Amorim, Nícia Rosário-Ferreira, Irina S. Moreira"
__email__ = "irina.moreira@cnc.uc.pt"
__group__ = "Data-Driven Molecular Design"
__group_leader__ = "Irina S. Moreira"
__project__ = "GPCR-A17 MAAP: Mapping Modulators, Agonists, and Antagonists to Predict the Next Bioactive Target"
"""

import json
import pandas as pd
import numpy as np
import random
import joblib  # For saving the model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score, confusion_matrix, cohen_kappa_score, classification_report
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from variables import *
   
# Set a global seed for reproducibility
GLOBAL_SEED = 42

# Fix seeds for Python's random, numpy, and any potential randomness in LightGBM
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)

# Load the data
Xtraintest = pd.read_csv(SPLITS_FOLDER + SYSTEM_SEP + "X_GPCRA17_Mold_PT.csv")
Xval = pd.read_csv(SPLITS_FOLDER + SYSTEM_SEP + "X_DNS_Mold_PT.csv")
ytraintest = pd.read_csv(SPLITS_FOLDER + SYSTEM_SEP + "y_GPCRA17_Mold_PT.csv")
yval = pd.read_csv(SPLITS_FOLDER + SYSTEM_SEP + "y_DNS_Mold_PT.csv")

# Split the data into training, testing, and validation sets
Xtrain, Xtemp, ytrain, ytemp = train_test_split(Xtraintest, ytraintest, train_size=0.80, random_state=GLOBAL_SEED)
X_val, Xtest, y_val, ytest = train_test_split(Xtemp, ytemp, test_size=0.5, random_state=GLOBAL_SEED)

# Standardize the data
scaler = StandardScaler()
SX_train = scaler.fit_transform(Xtrain)
SX_test = scaler.transform(Xtest)
SXval = scaler.transform(Xval)
SX_val = scaler.transform(X_val)

# Convert labels to numpy arrays and flatten them (no one-hot encoding)
ytrain_np = np.asarray(ytrain).ravel()
ytest_np = np.asarray(ytest).ravel()
yval_np = np.asarray(yval).ravel()
y_val_np = np.asarray(y_val).ravel()

# DNN model with optimized hyperparameters
# Load the saved parameters from the JSON file
with open(HYPERPARAMETERS_FOLDER + SYSTEM_SEP +'best_hyperparameters_DNN.json', 'r') as f:
    loaded_params = json.load(f)

# Helper function to calculate specificity from confusion matrix
def calculate_specificity(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    TN = cm.diagonal().sum() - cm.sum(axis=0) + cm.diagonal()  # TN = sum of true negatives
    FP = cm.sum(axis=0) - cm.diagonal()
    specificity = TN / (TN + FP)
    return specificity.mean()

# Helper function to calculate all metrics
def calculate_metrics(X, y, model):
    y_pred_prob = model.predict(X)  # Predict probabilities directly from the model
    y_pred = np.argmax(y_pred_prob, axis=1)  # Convert probabilities to class labels

    metrics = {
        "f1_score_weighted": f1_score(y, y_pred, average='weighted'),
        "precision_weighted": precision_score(y, y_pred, average='weighted'),
        "recall_weighted": recall_score(y, y_pred, average='weighted'),
        "accuracy": accuracy_score(y, y_pred),
        "specificity": calculate_specificity(y, y_pred),
        "cohens_kappa": cohen_kappa_score(y, y_pred),
    }

    # AUC requires the predicted probabilities and true labels
    try:
        auc = roc_auc_score(y, y_pred_prob, multi_class='ovr', average='weighted')
        metrics["auc"] = auc
    except ValueError as e:
        print(f"AUC computation error: {e}")
        metrics["auc"] = None  # Handle cases where AUC can't be computed
    
    # Get precision, recall, and f1 scores for each class
    class_metrics = classification_report(y, y_pred, output_dict=True)
    metrics["class_metrics"] = class_metrics
    return metrics

# Define the neural network creation function
def create_keras_model(n_layers, nodes, dropouts, l2_regs, learning_rate):
    model = Sequential()
    for i in range(n_layers):
        if i == 0:
            model.add(Dense(nodes[i], activation='relu', input_shape=(SX_train.shape[1],),
                            kernel_regularizer=l2(l2_regs[i])))
        else:
            model.add(Dense(nodes[i], activation='relu', kernel_regularizer=l2(l2_regs[i])))
        model.add(Dropout(dropouts[i]))
    model.add(Dense(3, activation='softmax'))  # 3 output neurons for 3 classes
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Create the model based on the loaded parameters
best_model = create_keras_model(
    loaded_params['n_layers'],
    [loaded_params[f'nodes_l{i}'] for i in range(loaded_params['n_layers'])],
    [loaded_params[f'dropout_l{i}'] for i in range(loaded_params['n_layers'])],
    [loaded_params[f'l2_l{i}'] for i in range(loaded_params['n_layers'])],
    loaded_params['learning_rate']
)

# Set early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Initialize lists to store metrics for multiple runs
metrics_train = []
metrics_val = []
metrics_test = []

# Variable to keep track of the best model based on validation F1 score
best_model = None
best_f1_score = 0
best_run = None

# Number of random initializations
n_runs = 5

# Perform multiple runs with different random seeds
for run in range(n_runs):
    # Set the seed for this run (use GLOBAL_SEED to ensure reproducibility)
    np.random.seed(GLOBAL_SEED + run)
    loaded_params['random_state'] = GLOBAL_SEED + run
    
    # Create the model based on the loaded parameters
    best_model = create_keras_model(
        loaded_params['n_layers'],
        [loaded_params[f'nodes_l{i}'] for i in range(loaded_params['n_layers'])],
        [loaded_params[f'dropout_l{i}'] for i in range(loaded_params['n_layers'])],
        [loaded_params[f'l2_l{i}'] for i in range(loaded_params['n_layers'])],
        loaded_params['learning_rate']
    )

    # Set early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Train the model (no one-hot encoding, using sparse categorical crossentropy)
    history = best_model.fit(SX_train, ytrain_np, 
                             validation_data=(SX_val, y_val_np),
                             epochs=loaded_params['epochs'], 
                             batch_size=loaded_params['batch_size'], 
                             callbacks=[early_stopping],
                             verbose=1)
    
    # Collect metrics for training, validation, and test sets
    train_metrics = calculate_metrics(SX_train, ytrain_np, best_model)
    test_metrics = calculate_metrics(SX_test, ytest_np, best_model)
    val_metrics = calculate_metrics(SXval, yval_np, best_model)
    
    metrics_train.append(train_metrics)
    metrics_test.append(test_metrics)
    metrics_val.append(val_metrics)

    # Check if this is the best model based on validation F1 score
    if val_metrics['f1_score_weighted'] > best_f1_score:
        best_f1_score = val_metrics['f1_score_weighted']
        best_model = best_model
        best_run = run

# Save the best model using joblib
joblib.dump(best_model, MODEL_FOLDER + SYSTEM_SEP + f'best_DNN_model_noki_run_{best_run}.joblib')

# Convert lists to numpy arrays for easy calculation of mean and std deviation
metrics_train = np.array(metrics_train)
metrics_test = np.array(metrics_test)
metrics_val = np.array(metrics_val)

# Function to print mean and std of metrics for training, test, and validation sets
def print_mean_std(metrics, dataset_name):
    print(f"\n{dataset_name} Metrics (mean ± std):")
    for key in metrics[0].keys():
        if key == "class_metrics":
            continue  # Skip class-specific metrics in the main summary
        values = [m[key] for m in metrics]
        mean_val = np.mean(values)
        std_val = np.std(values)
        print(f"{key}: {mean_val:.4f} ± {std_val:.4f}")

# Print the average and standard deviation of the metrics across the runs for each dataset
print_mean_std(metrics_train, "Training Set")
print_mean_std(metrics_test, "Test Set")
print_mean_std(metrics_val, "Drug Validation Set")

# Function to calculate mean and std for each class and metric
def calculate_class_metrics_mean_std(metrics_list, metric_type):
    class_metrics = [m["class_metrics"] for m in metrics_list]
    classes = class_metrics[0].keys()  # Get the class labels
    
    mean_metrics = {}
    std_metrics = {}
    
    for cls in classes:
        if cls not in ['accuracy', 'macro avg', 'weighted avg']:
            metric_values = [m[cls][metric_type] for m in class_metrics]
            mean_metrics[cls] = np.mean(metric_values)
            std_metrics[cls] = np.std(metric_values)
    
    return mean_metrics, std_metrics

# Calculate mean and std for precision, recall, and f1 score for each class for Drug Validation Set
precision_mean, precision_std = calculate_class_metrics_mean_std(metrics_val, "precision")
recall_mean, recall_std = calculate_class_metrics_mean_std(metrics_val, "recall")
f1_mean, f1_std = calculate_class_metrics_mean_std(metrics_val, "f1-score")

# Print out the class-specific mean and std for each metric (Drug Validation Set)
print("\nPrecision (mean ± std) for each class (Drug Validation Set):")
for cls in precision_mean:
    print(f"Class {cls}: {precision_mean[cls]:.4f} ± {precision_std[cls]:.4f}")

print("\nRecall (mean ± std) for each class (Drug Validation Set):")
for cls in recall_mean:
    print(f"Class {cls}: {recall_mean[cls]:.4f} ± {recall_std[cls]:.4f}")

print("\nF1 Score (mean ± std) for each class (Drug Validation Set):")
for cls in f1_mean:
    print(f"Class {cls}: {f1_mean[cls]:.4f} ± {f1_std[cls]:.4f}")

# After identifying and saving the best model, calculate the classification report
y_pred_val_prob = best_model.predict(SXval)  # Use the best model to predict on the Drug Validation Set

# Convert predicted probabilities to class labels
y_pred_val = np.argmax(y_pred_val_prob, axis=1)  # Get the class with the highest probability

# Calculate the classification report for the Drug Validation Set
classification_report_val = classification_report(yval_np, y_pred_val)

# Print the classification report for the Drug Validation Set
print("\nClassification Report for the Drug Validation Set (Best Model):")
print(classification_report_val)

# Save the best model's classification report and performance
print(f"\nBest model found in run {best_run} with F1 score: {best_f1_score:.4f}")
