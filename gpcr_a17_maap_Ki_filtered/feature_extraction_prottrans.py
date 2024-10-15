# -*- coding: utf-8 -*-

"""
__author__ = "Ana B. Caniceiro, Ana M. B. Amorim, Nícia Rosário-Ferreira, Irina S. Moreira"
__email__ = "irina.moreira@cnc.uc.pt"
__group__ = "Data-Driven Molecular Design"
__group_leader__ = "Irina S. Moreira"
__project__ = "GPCR-A17 MAAP: Mapping Modulators, Agonists, and Antagonists to Predict the Next Bioactive Target"
"""

from transformers import T5Tokenizer, T5EncoderModel
import torch
import re
import pandas as pd
from variables import *

# Load tokenizer and model
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False)
model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc").to(device)

# Set model precision
if device == torch.device("cpu"):
    model.to(torch.float32)

# Define file path
file_path = DATA_FOLDER + SYSTEM_SEP + "Receptor_sequences.fasta" 
output_path = PROTTRANS_FOLDER + SYSTEM_SEP 

# Function to read sequences and receptor names from a FASTA file
def read_fasta(file_path):
    sequences = []
    receptor_names = []
    with open(file_path, 'r') as file:
        sequence = ""
        for line in file:
            line = line.strip()
            if line.startswith(">"):
                if sequence:
                    sequences.append(sequence)
                    sequence = ""
                receptor_names.append(line[1:])
            else:
                sequence += line
        if sequence:
            sequences.append(sequence)
    return receptor_names, sequences

# Read sequences and receptor names from the file
receptor_names, sequences = read_fasta(file_path)

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

    # If you want to derive a single representation (per-protein embedding) for the whole protein
    protein_embedding = sequence_embedding.mean(dim=0)  # Shape (1024)

    # Collect the embedding for this protein
    all_protein_embeddings.append(protein_embedding.cpu().numpy())

# Create a DataFrame with sequences and embeddings
embedding_columns = [f'Embedding_{i+1}' for i in range(all_protein_embeddings[0].shape[0])]
data = {'Receptor': receptor_names}
for idx, embedding in enumerate(all_protein_embeddings):
    for i, value in enumerate(embedding):
        if f'Embedding_{i+1}' not in data:
            data[f'Embedding_{i+1}'] = []
        data[f'Embedding_{i+1}'].append(value)

df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
df.to_csv(output_path + "ProtTrans_embeddings.csv", index=False)

# Display the DataFrame
print(df)