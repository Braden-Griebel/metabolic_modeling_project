"""
Combining data from the model, uniprot, and protein properties, as well as protein phosphorylation and acetylation data
"""

# Setup
# Standard Library Imports
import pathlib

# External imports
import numpy as np
import pandas as pd

# Local Imports

# Path Setup
BASE_PATH = pathlib.Path(__file__).parent.parent
DATA_PATH = BASE_PATH / "data"
RESULTS_PATH = BASE_PATH / "results"

# Read in feature data
model_data = pd.read_csv(
    RESULTS_PATH / "model_based_features.csv", index_col=0
)  # Index column is Rv
protein_data = pd.read_csv(
    RESULTS_PATH / "protein_info.csv", index_col=0
)  # Index column is Rv

# Read in target data
kinase_list = [
    "PknB",
    "PknD",
    "PknE",
    "PknF",
    "PknG",
    "PknH",
    "PknI",
    "PknJ",
    "PknK",
    "PknL",
]
kinase_targets = {}
for kinase in kinase_list:
    df = pd.read_excel(
        DATA_PATH / "frando_phos_data.xlsx", sheet_name=kinase, header=0, index_col=None
    )
    kinase_targets[kinase] = set(df[df["Indirect?"] != "Indirect"]["Rv Number"])

# Read in acetylation data
acetylation_targets = pd.read_excel(
    DATA_PATH / "xie_acetylation.xlsx", sheet_name="SWNU_Mtu_Kac", index_col=0, header=0
)  # Index is Locus number

# Join the model data and the protein info dataframes 
combined_df = pd.merge(model_data, protein_data, left_index=True, right_index=True)
print(combined_df)
