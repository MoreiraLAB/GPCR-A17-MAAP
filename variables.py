# -*- coding: utf-8 -*-
"""
__author__ = "Ana B. Caniceiro, Ana M. B. Amorim, Nícia Rosário-Ferreira, Irina S. Moreira"
__email__ = "irina.moreira@cnc.uc.pt"
__group__ = "Data-Driven Molecular Design"
__group_leader__ = "Irina S. Moreira"
__project__ = "GPCR-A17 MAAP: Mapping Modulators, Agonists, and Antagonists to Predict the Next Bioactive Target"
"""

# After cloning, to run this repository, change the DEFAULT_LOCATION 
# to the directory where this repository is cloned.

#Folder locations
SYSTEM_SEP = "/"
DEFAULT_LOCATION = r"/change/to/your/directory/GPCR-A17-MAAP-main" #Change here to your working directory
DATA_FOLDER = DEFAULT_LOCATION + SYSTEM_SEP + "data"
FEATURES_FOLDER = DEFAULT_LOCATION + SYSTEM_SEP + "features"
MOLD2_FOLDER = FEATURES_FOLDER + SYSTEM_SEP + "mold2"
PROTTRANS_FOLDER = FEATURES_FOLDER + SYSTEM_SEP + "prottrans"
SPLITS_FOLDER = FEATURES_FOLDER + SYSTEM_SEP + "splits"
MODEL_FOLDER = DEFAULT_LOCATION + SYSTEM_SEP + "models"
HYPERPARAMETERS_FOLDER = DEFAULT_LOCATION + SYSTEM_SEP + "best_hyperparameters"
MODEL_NOKI_FOLDER = DEFAULT_LOCATION + SYSTEM_SEP + "gpcr_a17_maap_ki_excluded" + SYSTEM_SEP + "models"
DATAFRAMES_FOLDER = DEFAULT_LOCATION + SYSTEM_SEP + "dataframes"
