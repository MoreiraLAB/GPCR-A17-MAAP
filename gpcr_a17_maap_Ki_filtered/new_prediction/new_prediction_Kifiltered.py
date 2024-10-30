# -*- coding: utf-8 -*-
"""
__author__ = "Ana B. Caniceiro, Ana M. B. Amorim, Nícia Rosário-Ferreira, Irina S. Moreira"
__email__ = "irina.moreira@cnc.uc.pt"
__group__ = "Data-Driven Molecular Design"
__group_leader__ = "Irina S. Moreira"
__project__ = "GPCR-A17 MAAP: Mapping Modulators, Agonists, and Antagonists to Predict the Next Bioactive Target"
"""

import pandas as pd
from rdkit import Chem
import joblib
from transformers import T5Tokenizer, T5EncoderModel
import torch
import re
from Mold2_pywrapper import Mold2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import random
import numpy as np
from variables_new_pred import *

# Set a global seed for reproducibility
GLOBAL_SEED = 42

# Fix seeds for Python's random, numpy, and any potential randomness in LightGBM
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)

# Load the data
Xtraintest = pd.read_csv(DATA_FOLDER + SYSTEM_SEP + "X_GPCRA17_ki_filtered_Mold_PT.csv")
Xval = pd.read_csv(DATA_FOLDER + SYSTEM_SEP + "X_DNS_ki_filtered_Mold_PT.csv")
ytraintest = pd.read_csv(DATA_FOLDER + SYSTEM_SEP + "y_GPCRA17_ki_filtered_Mold_PT.csv")
yval = pd.read_csv(DATA_FOLDER + SYSTEM_SEP + "y_DNS_ki_filtered_Mold_PT.csv")

# Split the data into training, testing, and validation sets
Xtrain, Xtemp, ytrain, ytemp = train_test_split(Xtraintest, ytraintest, train_size=0.80, random_state=GLOBAL_SEED)
X_val, Xtest, y_val, ytest = train_test_split(Xtemp, ytemp, test_size=0.5, random_state=GLOBAL_SEED)

# Function to uniformize SMILES
def standardize_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        standardized_smiles = Chem.MolToSmiles(mol)
        return standardized_smiles
    else:
        return None

# Extract ProtTrans features from sequences
# Load tokenizer and model
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False)
model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc").to(device)

# Set model precision if on CPU
if device == torch.device("cpu"):
    model.to(torch.float32)

# Define  new_prediction file path 
input_excel_path = DATA_FOLDER + SYSTEM_SEP + "new_prediction.xlsx"  # Input CSV containing sequence column

#Load excel file
df = pd.read_excel(input_excel_path)

# Read sequences from new_prediction excel file
sequences = df['sequence'].tolist()  # Column for sequences

# Initialize list to collect per-protein embeddings
all_protein_embeddings = []

# Process each sequence
for sequence in sequences:
    # Replace rare/ambiguous amino acids by X and introduce white-space between all amino acids
    sequence = " ".join(list(re.sub(r"[UZOB]", "X", sequence)))

    # Tokenize sequence and pad up to the longest sequence in the batch
    ids = tokenizer(sequence, add_special_tokens=True, padding="longest")
    input_ids = torch.tensor(ids['input_ids']).to(device)
    attention_mask = torch.tensor(ids['attention_mask']).to(device)

    # Generate embeddings
    with torch.no_grad():
        embedding_repr = model(input_ids=input_ids.unsqueeze(0), attention_mask=attention_mask.unsqueeze(0))

    # Extract residue embeddings for the sequence and remove padded & special tokens
    sequence_length = len(sequence.split())
    sequence_embedding = embedding_repr.last_hidden_state[0, :sequence_length]

    # Derive a single representation (per-protein embedding) for the whole protein
    protein_embedding = sequence_embedding.mean(dim=0)  # Shape (1024)

    # Collect the embedding for this protein
    all_protein_embeddings.append(protein_embedding.cpu().numpy())

# Create a DataFrame with embeddings only
embedding_columns = [f'Embedding_{i+1}' for i in range(all_protein_embeddings[0].shape[0])]
data = {}
for idx, embedding in enumerate(all_protein_embeddings):
    for i, value in enumerate(embedding):
        if f'Embedding_{i+1}' not in data:
            data[f'Embedding_{i+1}'] = []
        data[f'Embedding_{i+1}'].append(value)

df_embeddings = pd.DataFrame(data)


# Extract Mold2 features from SMILES
# Extract the 'smile' column, drop missing values, and convert to a list of strings
Smiles_list = df['smile'].dropna().astype(str).tolist()

# Convert SMILES strings to RDKit molecules, handling invalid SMILES
valid_smiles = []
mols = []
for smiles in Smiles_list:
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        mols.append(mol)
        valid_smiles.append(smiles)  # Keep track of valid SMILES
    else:
        print(f"Invalid SMILES string: {smiles}")

# Initialize Mold2 descriptor calculator
mold2 = Mold2()

# Calculate descriptors using Mold2
descriptors = mold2.calculate(mols)

# Convert the descriptors to a pandas DataFrame
descriptors_df = pd.DataFrame(descriptors)

# Add the valid SMILES strings as a new column in the DataFrame
descriptors_df['smile'] = valid_smiles

# Join embeddings and Mold2 features
features = pd.concat([descriptors_df, df_embeddings], axis=1)

# At this point, features contains both the Mold2 features and the ProtTrans embeddings.

# Now, we'll filter this DataFrame to keep only the columns that are present in X_train.
# Xtrain.columns gives the list of columns we want to keep.

# List of columns to keep (present in X_train)
columns_to_keep = Xtrain.columns.intersection(features.columns)

# Select only the columns present in X_train
filtered_features = features[columns_to_keep]

# Now, filtered_features contains only the columns that are present in X_train.

# Make a copy of filtered_features to avoid modifying a slice of the original DataFrame
filtered_features = filtered_features.copy()

# Read the Ki values from the 'Ki' column of the new_prediction Excel file
ki_values = df['Ki']  # Ki column

# Add the Ki column to the filtered DataFrame
filtered_features.loc[:, 'Ki'] = ki_values

# Ensure that 'Ki' is the first column
# Reorder the columns with 'Ki' first, followed by the remaining columns
columns = ['Ki'] + [col for col in filtered_features.columns if col != 'Ki']
filtered_features = filtered_features[columns]

# Save the resulting DataFrame to a CSV file
filtered_features.to_csv(FEATURES_FOLDER + SYSTEM_SEP + "ProtTrans_Mold2_Ki_features.csv", index=False)

# Standardize the data
scaler = StandardScaler()
SX_train = scaler.fit_transform(Xtrain)  # Fitting the scaler on training data
SX_new_pred = scaler.transform(filtered_features)  # Applying the same scaler to the new data (filtered_features)

# Load ML models
# Load best XGBoost base model
XBG_model = joblib.load(MODEL_FOLDER + SYSTEM_SEP + 'best_ki_filtered_xgboost_model_run_3.joblib')
# Load best LightGBM base model
LightGBM_model = joblib.load(MODEL_FOLDER + SYSTEM_SEP + 'best_ki_filtered_lgbm_model_run_1.joblib')    
# Load best Random Forest base model
RF_model = joblib.load( MODEL_FOLDER + SYSTEM_SEP + 'best_ki_filtered_rf_model_run_3.joblib')

# Construction of ensemble method (Blending approach)
# Make predictions on the new prediction dataframe features with the base models
rf_preds_new = RF_model.predict_proba(SX_new_pred)
lightgbm_preds_new = LightGBM_model.predict_proba(SX_new_pred)
xgb_preds_new = XBG_model.predict_proba(SX_new_pred)

# Combine original features with predicted probabilities (SX_new_pred + 3 columns with probabilities for each class from each base model)
meta_features_new = np.hstack((SX_new_pred, rf_preds_new, lightgbm_preds_new, xgb_preds_new))

# Load the pre-trained GPCR_A17_MAAP model
gpcr_a17_maap = joblib.load(MODEL_FOLDER + SYSTEM_SEP + 'best_ki_filtered_meta_model_run_3.joblib')

# Use the model to make predictions on the standardized new prediction data + class probabilities (meta_features_new)
new_predictions = gpcr_a17_maap.predict(meta_features_new)

# Add the predictions as a new column to the filtered_features DataFrame
filtered_features['Predictions'] = new_predictions

# Create a new DataFrame with just the Predictions column
predictions_df = filtered_features[['Predictions']]

# Save the Predictions DataFrame to a CSV file
predictions_df.to_csv("predictions_GPCRA17_MAAP_Kifiltered.csv", index=False)

# Optionally, print or return the Predictions DataFrame
print(predictions_df.head())