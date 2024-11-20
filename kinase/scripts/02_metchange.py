"""
Determine the inconsistency score for all kinases and wild type samples
"""

# Setup
# Imports
# Standard Library Imports
import logging
import pathlib
import warnings

# External Imports
import metworkpy
import pandas as pd

# Local Imports

# Setup Path
BASE_PATH = pathlib.Path(__file__).parent.parent
DATA_PATH = BASE_PATH / "data"
RESULTS_PATH = BASE_PATH / "results"
MODEL_PATH = BASE_PATH.parent / "models"


# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    filename=BASE_PATH / "diagnostics" / "02_metchange.log",
    filemode="w",
    level=logging.INFO,
)

# Import Base Model
logger.info("Importing Base Model")
BASE_MODEL = metworkpy.read_model(MODEL_PATH / "iEK1011_m7H10_media.json")

# Run Parameters
PROCESSES = 12
QUANTILE_CUTOFF = 0.2
GENE_SUBSET = BASE_MODEL.genes.list_attr("id")

# Read in the expression data
logger.info("Reading in expression data")
expr_rpkm = pd.read_csv(
    DATA_PATH / "frando_gene_expr_rpkm.csv",
    index_col=0,
    header=0,
    usecols=lambda x: x not in ["GrowthPhase"],
).drop(["WT10", "WT11"])

# Create Results dataframe
metchange_results = pd.DataFrame(
    0.0,
    index=pd.Index(expr_rpkm.index),
    columns=pd.Index(BASE_MODEL.metabolites.list_attr("id")),
)


for sample in expr_rpkm.index:
    logger.info(f"Starting sample {sample}")
    # Get expression series
    expr_ser = expr_rpkm.loc[sample, :]
    # Convert expression to gene weights
    logger.info("Converting expression to gene weights")
    gene_weights = metworkpy.utils.expr_to_metchange_gene_weights(
        expression=expr_ser, quantile_cutoff=QUANTILE_CUTOFF, subset=GENE_SUBSET
    )
    # Convert gene weights to reaction weights
    logger.info("Covnerting gene weights to reaction weights")
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Genes .* are in model but not in gene weights, setting their weights to .*",
            category=UserWarning,
            module=r".*metworkpy.*",
        )
        rxn_weights = metworkpy.gpr.gene_to_rxn_weights(
            model=BASE_MODEL,
            gene_weights=gene_weights,
            fn_dict=metworkpy.gpr.gpr_functions.METCHANGE_FUNC_DICT,
            fill_val=0,
        )
    # Run metchange
    logger.info("Starting Metchange")
    metchange_tmp_res = metworkpy.metabolites.metchange(
        model=BASE_MODEL, reaction_weights=rxn_weights, metabolites=None
    )
    # Save metchange results to DF
    logger.info("Saving metchange results to dataframe")
    metchange_results.loc[sample, metchange_tmp_res.index] = metchange_tmp_res

# Save metchange results
logger.info("Saving metchange results to disk")
metchange_results.to_csv(RESULTS_PATH / "metchange_results.csv")
