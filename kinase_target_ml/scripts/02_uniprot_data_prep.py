"""
Script to clean up Uniprot data for the different genes
"""

# Setup
# Standard Library Imports
import pathlib

# External Imports
import numpy as np
import pandas as pd

# Local Imports

# Path Setup
BASE_PATH = pathlib.Path(__file__).parent.parent
DATA_PATH = BASE_PATH / "data"

# Read in Uniprot Data
uniprot_df = pd.read_csv(
    DATA_PATH / "uniprotkb_taxonomy_id_83332_2025_03_03.tsv", sep="\t"
)

# Rename the Ordered Locus column to Locus
uniprot_df = uniprot_df.rename({"Gene Names (ordered locus)": "Locus"}, axis=1)

# Expand the ordered locus columns to only have a single locus per row
uniprot_df["Locus"] = uniprot_df["Locus"].str.split(r" |;|/|,")
uniprot_df = uniprot_df.explode("Locus")

# For all the rows where the locus is an empty string, change it to be a NaN
uniprot_df = uniprot_df.replace("", np.nan)

# Filter for columns of interest
uniprot_df = uniprot_df[["Locus", "Length", "Mass", "Sequence"]]

# Drop all the rows with nan
uniprot_df = uniprot_df.dropna(axis=0)

# Save the filtered uniprot_df
uniprot_df.to_csv(DATA_PATH / "uniprot_cleaned.csv", index=False)
