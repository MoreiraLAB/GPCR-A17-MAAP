# -*- coding: utf-8 -*-

"""
__author__ = "Ana B. Caniceiro, Ana M. B. Amorim, Nícia Rosário-Ferreira, Irina S. Moreira"
__email__ = "irina.moreira@cnc.uc.pt"
__group__ = "Data-Driven Molecular Design"
__group_leader__ = "Irina S. Moreira"
__project__ = "GPCR-A17 MAAP: Mapping Modulators, Agonists, and Antagonists to Predict the Next Bioactive Target"
"""

import pandas as pd
from variables import *

#Split of the dataset for training and testing and the dataset with drugs never seen in X and y

# Load datasets
Dataset_final = pd.read_csv(FEATURES_FOLDER + SYSTEM_SEP + 'GPCRA17_features_Mold2_PT.csv', low_memory=False)  
DNS_dataset = pd.read_csv(FEATURES_FOLDER + SYSTEM_SEP + 'GPCRA17_DNS_features_Mold2_PT.csv', low_memory=False)

# Format the dataset for ML
# Transform 'Action' column into 'Class' with categorical values
replacements = {'Agonist': 1,'Antagonist': 0,'Modulator': 2}

Dataset_final['Class'] = Dataset_final['Action'].replace(replacements)
DNS_dataset['Class'] = DNS_dataset['Action'].replace(replacements)

# Drop the original 'Action' column
Dataset_final = Dataset_final.drop(columns=['Action'])
DNS_dataset = DNS_dataset.drop(columns=['Action'])


# Separate features and labels for original and validation datasets
X = Dataset_final.drop(columns =['Receptor', 'Smiles', 'Ligand', 'Class'], axis=1)
y = Dataset_final['Class']

X_DNS = DNS_dataset.drop(columns =['Receptor', 'Smiles', 'Ligand', 'Class'], axis=1)
y_DNS = DNS_dataset['Class']

# Save the processed datasets to CSV files
X.to_csv(SPLITS_FOLDER + SYSTEM_SEP + 'X_GPCRA17_Mold_PT.csv', index=False)
y.to_csv(SPLITS_FOLDER + SYSTEM_SEP + 'y_GPCRA17_Mold_PT.csv', index=False)
X_DNS.to_csv(SPLITS_FOLDER + SYSTEM_SEP + 'X_DNS_Mold_PT.csv', index=False)
y_DNS.to_csv(SPLITS_FOLDER + SYSTEM_SEP + 'y_DNS_Mold_PT.csv', index=False)