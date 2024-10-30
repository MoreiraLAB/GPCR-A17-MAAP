# -*- coding: utf-8 -*-

"""
__author__ = "Ana B. Caniceiro, Ana M. B. Amorim, Nícia Rosário-Ferreira, Irina S. Moreira"
__email__ = "irina.moreira@cnc.uc.pt"
__group__ = "Data-Driven Molecular Design"
__group_leader__ = "Irina S. Moreira"
__project__ = "GPCR-A17 MAAP: Mapping Modulators, Agonists, and Antagonists to Predict the Next Bioactive Target"
"""

# 1. Join features from Mold2 (ligands' features) and ProtTrans (Protein embeddings)

import pandas as pd
from variables import *
import os 

# Load datasets and features
# Change for your files names
GPCR_A17_df= pd.read_csv(DATA_FOLDER + SYSTEM_SEP + 'GPCRA17_Kifiltered.csv')
ProtTrans_features_df =pd.read_csv(PROTTRANS_FOLDER + SYSTEM_SEP + 'ProtTrans_embeddings.csv', low_memory=False)
Mold_features_df=pd.read_csv(MOLD2_FOLDER + SYSTEM_SEP + 'GPCRA17_features_Mold2.csv', low_memory=False)

# Ensure SMILES column is of type string
GPCR_A17_df['Smiles'] = GPCR_A17_df['Smiles'].astype(str)

# Merge datasets on 'Smiles' and 'Receptor' columns
GPCR_Mold_df = pd.merge(GPCR_A17_df, Mold_features_df, on='Smiles', how='outer')

GPCR_Mold_PT_df = pd.merge(GPCR_Mold_df, ProtTrans_features_df, on='Receptor', how='outer')

GPCR_Mold_PT_df = GPCR_Mold_PT_df.dropna()


## 2. Creation of a validation datatset with ligands that only appear in drugs never seen dataset

#validation -10% of entire dataset

#the validation should be unique entries, that is, protein-ligand complexes, where the ligand only binds to only one protein
#10% of 4274 is 427 entries
#427 entries for validation dataset

# Remove duplicate entries from the dataset based on the 'Smiles' column.
# Only keeps rows where 'Smiles' are unique (drop rows that have duplicate 'Smiles' values).
deduplicated_data = GPCR_Mold_PT_df.drop_duplicates(subset=['Smiles'], keep=False)

# Check if there are at least 427 unique entries after deduplication.
# If there are, take a random sample of 427 rows from the deduplicated data.
# If your dataset has more entries, please change the number of unique entries.

if deduplicated_data.shape[0] >= 427:
    # Randomly sample 427 entries from the deduplicated dataset, using a fixed random seed for reproducibility.
    test_data = deduplicated_data.sample(n=427, random_state=42)
else:
    # If there are fewer than 427 unique entries, print an error message.
    print("Error: Not enough entries to sample 427.")

# Get the indices of the test data.
Test_indices = test_data.index

# Remove the sampled test data from the main dataset to create a new dataset without those entries.
GPCR_Mold_PT_df_withoutval = GPCR_Mold_PT_df.drop(Test_indices)

# Change directory to save the final validation set
output_dir = FEATURES_FOLDER + SYSTEM_SEP
os.chdir(output_dir)

# Save the validation set and the combined dataset
test_data.to_csv('GPCRA17_DNS_ki_filtered_features_Mold2_PT.csv', index=False)
GPCR_Mold_PT_df_withoutval.to_csv('GPCRA17_ki_filtered_features_Mold2_PT.csv', index=False)