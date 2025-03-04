"""
Script for generating IMAT models from the Transcription Factor Overexpression Data
"""

# Setup
# Imports
# Standard Library Imports
import itertools
import logging
import pathlib
from typing import cast
import warnings

# External Imports
from cobra.core import gene
import numpy as np
import metworkpy
import pandas as pd

# Local Imports

# Setup Path
BASE_PATH = pathlib.Path(__file__).parent.parent
DATA_PATH = BASE_PATH / "data"
RESULTS_PATH = BASE_PATH / "results"
MODEL_PATH = BASE_PATH.parent / "models"
MODEL_OUT_PATH = BASE_PATH / "tfoe_models"

# Setup Logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    filename=BASE_PATH / "diagnostics" / "01_model_generation.log",
    filemode="a",
    level=logging.INFO,
)

# Read in base model
logger.info("Reading in m7H10 model")
BASE_MODEL = metworkpy.read_model(MODEL_PATH / "iEK1011_m7H10_media.json")

# Run Parameters
PROCESSES = 12
QUANTILE = (0.15, 0.85)
EPSILON = 1.0
THRESHOLD = 0.01
GENE_SUBSET = BASE_MODEL.genes.list_attr("id")
NEG_FOLD_CHANGE = -1
POS_FOLD_CHANGE = 1
OBJECTIVE_TOLERANCE = 5e-2

# Read in the expression data
logger.info("Reading in TFOE expression data")
tfoe_expr_df = pd.read_csv(DATA_PATH / "GSE59086_tfoe.csv", index_col=0, header=0)
tfoe_expr_df.index = tfoe_expr_df.index.str.replace("\n", "")
tfoe_expr_df.loc["wildtype"] = tfoe_expr_df.median(axis=0)

# Create the Models using the quantile method
for sample in tfoe_expr_df.index:
    logger.info(f"Starting {sample} quant model generation")
    model_out_path = MODEL_OUT_PATH / "quant_models" / f"{sample}_quant.json"
    if model_out_path.exists():
        continue  # Model already Generated
    expr_series = tfoe_expr_df.loc[sample]
    # Convert gene expression to gene weights
    logger.info("Converting gene expression to gene weights")
    gene_weights = metworkpy.utils.expr_to_imat_gene_weights(
        expression=expr_series,
        quantile=QUANTILE,
        subset=GENE_SUBSET,
    )
    # Convert gene weights to reaction weights
    logger.info("Converting gene weights to reaction weights")
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Genes .* are in model but not int gene weights.*",
            category=UserWarning,
            module=r".*metworkpy.*",
        )
        rxn_weights = metworkpy.gpr.gene_to_rxn_weights(
            model=BASE_MODEL,
            gene_weights=gene_weights,
            fn_dict=metworkpy.gpr.gpr_functions.IMAT_FUNC_DICT,
            fill_val=0.0,
        )
    # Generate the iMAT Model
    imat_model = metworkpy.imat.generate_model(
        model=BASE_MODEL,
        rxn_weights=rxn_weights,
        method="fva",
        epsilon=EPSILON,
        threshold=THRESHOLD,
        loopless=False,
        processes=PROCESSES,
        objective_tolerance=OBJECTIVE_TOLERANCE,
    )
    # Save model
    logger.info("Saving iMAT Model")
    metworkpy.write_model(model=imat_model, model_path=model_out_path)

# Next Generate the differential expression models
tfoe_median = tfoe_expr_df.loc["wildtype"]
tfoe_expr_df = tfoe_expr_df.drop("wildtype", axis=0)
tfoe_fc = tfoe_expr_df.div(tfoe_median, axis=1)
tfoe_l2fc = np.log2(tfoe_fc)
tfoe_l2fc = cast(pd.DataFrame, tfoe_l2fc)

for sample in tfoe_l2fc.index:
    logger.info(f"Starting {sample} diff")
    model_out_path = MODEL_OUT_PATH / "diff_models" / f"{sample}_diff.json"
    if model_out_path.exists():
        continue  # Model already generated
    # Convert log2fc to gene weights
    logger.info("Converting log2fc to gene weights")
    sample_l2fc = tfoe_l2fc.loc[sample]
    gene_weights = np.zeros(sample_l2fc.size)
    gene_weights[sample_l2fc <= NEG_FOLD_CHANGE] = -1
    gene_weights[sample_l2fc >= POS_FOLD_CHANGE] = 1
    gene_weights = pd.Series(gene_weights, index=sample_l2fc.index)
    # Convert gene weights to reaction weights
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Genes .* are in model but not int gene weights.*",
            category=UserWarning,
            module=r".*metworkpy.*",
        )
        rxn_weights = metworkpy.gpr.gene_to_rxn_weights(
            model=BASE_MODEL,
            gene_weights=gene_weights,
            fn_dict=metworkpy.gpr.gpr_functions.IMAT_FUNC_DICT,
            fill_val=0,
        )
    # Generate iMAT Model
    logger.info("Generating diff model")
    imat_model = metworkpy.imat.generate_model(
        model=BASE_MODEL,
        rxn_weights=rxn_weights,
        method="fva",
        epsilon=EPSILON,
        threshold=THRESHOLD,
        objective_tolerance=OBJECTIVE_TOLERANCE,
    )
    # Save the model
    logger.info("Saving model")
    metworkpy.write_model(model=imat_model, model_path=model_out_path)
