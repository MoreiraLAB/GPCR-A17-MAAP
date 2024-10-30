# -*- coding: utf-8 -*-

"""
__author__ = "Ana B. Caniceiro, Ana M. B. Amorim, Nícia Rosário-Ferreira, Irina S. Moreira"
__email__ = "irina.moreira@cnc.uc.pt"
__group__ = "Data-Driven Molecular Design"
__group_leader__ = "Irina S. Moreira"
__project__ = "GPCR-A17 MAAP: Mapping Modulators, Agonists, and Antagonists to Predict the Next Bioactive Target"
"""

import pandas as pd
from Mold2_pywrapper import Mold2
from rdkit import Chem
from variables import *

# Load the ligands data
Ligands = pd.read_csv(DATA_FOLDER + SYSTEM_SEP + 'ligands_A17_Kifiltered.csv')

# Define the results directory
results = MOLD2_FOLDER + SYSTEM_SEP

# Extract the 'Smiles' column, drop missing values, and convert to a list of strings
Smiles_list = Ligands['Smiles'].dropna().astype(str).tolist()

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
descriptors_df['Smiles'] = valid_smiles

feature_detail = mold2.descriptor_details()

# Calculation of variance of each feature
descriptors_df_2 =descriptors_df.drop(columns=['Smiles'])
variance_descriptors_per_column = descriptors_df_2.var()

# Print the variance of each column
print("Variance of each column descriptors:")
print(variance_descriptors_per_column)

# Filter columns where the variance is greater than 0
descriptors_var_above_0 = variance_descriptors_per_column[variance_descriptors_per_column > 0].index.tolist()
print("Columns with variance greater than 0:")
print(descriptors_var_above_0)

# Create a new DataFrame with columns having variance greater than 0 and add 'Smiles' column
descriptors_var_above_0_df = descriptors_df[descriptors_var_above_0].assign(Smiles=descriptors_df['Smiles'])

# Save the filtered descriptors DataFrame to a CSV file
descriptors_var_above_0_df.to_csv(results + 'GPCRA17_features_Mold2.csv', index=False)

# Get the names of the selected features
selected_features = descriptors_var_above_0_df.columns.tolist()

# Filter the descriptor details to include only the selected features
feature_detail_v0 = {k: v for k, v in feature_detail.items() if k in selected_features}

# Convert the filtered descriptor details to a DataFrame for better visualization
feature_detail_v0_df = pd.DataFrame.from_dict(feature_detail_v0, orient='index')

# Display the filtered descriptor details
print(feature_detail_v0_df)