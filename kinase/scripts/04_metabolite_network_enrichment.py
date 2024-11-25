"""
Find the Metabolite networks and determine if any of networks
are enriched for kinase targeted genes
"""

# Setup
# Imports
# Standard Library Imports
import pathlib

# External Imports
import metworkpy
import pandas as pd

# Local Imports

# Path Setup
BASE_PATH = pathlib.Path(__file__).parent.parent
RESULTS_PATH = BASE_PATH / "results"
DATA_PATH = BASE_PATH / "data"
MODEL_PATH = BASE_PATH.parent / "models"

# Read Base Model
BASE_MODEL = metworkpy.read_model(MODEL_PATH / "iEK1011_m7H10_media.json")

# Identify Metabolite Networks
