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

# Add in target columns (PknB-PknL, and Acetylation)
for kinase in kinase_list:
    # Create the column with all False
    combined_df[kinase] = False
    # Set the correct genes to True
    combined_df.loc[combined_df.index.isin(kinase_targets[kinase]), kinase] = True

# Add in the total kinase target
combined_df["phosphorylated"] = combined_df[kinase_list].any(axis=1)
# Add in the acetylation target
combined_df["acetylated"] = False
combined_df.loc[combined_df.index.isin(acetylation_targets.index), "acetylated"] = True

# Split the data into a features dataframe, and a targets dataframe
target_df = combined_df[kinase_list + ["phosphorylated", "acetylated"]]
feature_df = combined_df.drop(kinase_list + ["phosphorylated", "acetylated"], axis=1)

# Save the feature and target DataFrames to the Results folder
feature_df.to_csv(RESULTS_PATH / "feature_df.csv", index=True)
target_df.to_csv(RESULTS_PATH / "target_df.csv", index=True)
