"""
Script to generate IMAT models for each TF gene expression sample, and then perform flux sampling.
"""

# Setup
# Imports
# Standard Librar Imports
import logging
import pathlib
import warnings

# External Imports
from cobra.sampling import OptGPSampler
import metworkpy
import numpy as np
import pandas as pd

# Local Imports

# Setup Path
BASE_PATH = pathlib.Path(__file__).parent.parent
DATA_PATH = BASE_PATH / "data"
RESULTS_PATH = BASE_PATH / "results"
FLUX_SAMPLES_PATH = BASE_PATH / "flux_samples"
MODEL_PATH = BASE_PATH.parent / "models"
MODEL_OUT_PATH = BASE_PATH / "tfoe_models"

# Setup Logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    filename=BASE_PATH / "diagnostics" / "01_model_generation_and_sampling.log",
    filemode="w",
    level=logging.INFO,
)

# Read in the base model
BASE_MODEL = metworkpy.read_model(MODEL_PATH / "iEK1011_m7H10_media.json")

# Run Parameters
PROCESSES = 12
QUANTILE = (0.15, 0.85)
EPSILON = 1.0
THRESHOLD = 0.01
GENE_SUBSET = BASE_MODEL.genes.list_attr("id")
THINNING = 500
NUM_SAMPLES = 1_000

# Read in the expression data
logger.info("Reading in expression data")
expr = pd.read_csv(DATA_PATH / "GSE59086_tfoe.csv", index_col=0, header=0)
expr.index = expr.index.str.strip()
# Find the median across all samples to use as wildtype
expr_median = expr.median(axis="rows")
expr.loc["wt_median"] = expr_median

logger.info("Finding IMAT Models...")

for sample in expr.index:
    logger.info(f"Starting sample {sample}")
    outpath = MODEL_OUT_PATH / f"{sample}.json"
    if outpath.exists():
        continue
    # Filter for this sample's expression
    sample_expr = expr.loc[sample]
    # Find the gene weights
    logger.info("Converting gene expression to gene weights")
    gene_weights = metworkpy.utils.expr_to_imat_gene_weights(
        sample_expr, quantile=QUANTILE, subset=GENE_SUBSET
    )
    # Find the reaction weights
    logger.info("Converting gene weights to reaction weights")
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
            fn_dict=metworkpy.gpr.gpr_functions.IMAT_FUNC_DICT,
            fill_val=0,
        )
    logger.info("Generating IMAT FVA Model")
    imat_model = metworkpy.imat.generate_model(
        model=BASE_MODEL,
        rxn_weights=rxn_weights,
        method="fva",
        epsilon=EPSILON,
        threshold=THRESHOLD,
        loopless=False,
        processes=PROCESSES,
    )
    logger.info("Saving IMAT Model")
    metworkpy.write_model(model=imat_model, model_path=outpath)

# Generate differential expression models
logger.info("Generating models using differential expression")
expr_median = expr.loc["wt_median"]
expr_fold = expr.iloc[:-1]
expr_fold = expr_fold.div(expr_median, axis="columns")
expr_fold = np.log2(expr_fold)
for sample in expr_fold.index:
    logger.info(f"Starting model generation for {sample} using differential expression")
    outpath = MODEL_OUT_PATH / f"{sample}_diff.json"
    if outpath.exists():
        continue
    # Find gene weights
    logger.info("Finding gene weights")
    sample_expr = expr_fold.loc[sample]
    gene_weights = pd.Series(0.0, index=sample_expr.index)
    gene_weights[sample_expr <= -1.0] = -1.0
    gene_weights[sample_expr >= 1.0] = 1.0
    # Find reaction weights
    logger.info("Finding reaction weights")
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
            fn_dict=metworkpy.gpr.gpr_functions.IMAT_FUNC_DICT,
            fill_val=0,
        )
    logger.info("Generating model")
    imat_model = metworkpy.imat.generate_model(
        model=BASE_MODEL,
        rxn_weights=rxn_weights,
        method="fva",
        loopless=False,
        processes=PROCESSES,
        epsilon=EPSILON,
        threshold=THRESHOLD,
    )
    logger.info("Saving model")
    metworkpy.write_model(imat_model, outpath)
# Save the unconstrained model as wt_diff.json
metworkpy.write_model(model=BASE_MODEL, model_path=MODEL_OUT_PATH / "wt_diff.json")

# Now generate flux samples for all models
logger.info("Starting flux sampling")
for sample_path in MODEL_OUT_PATH.glob("*"):
    sample = sample_path.name.split(".")[0]
    logger.info(f"Starting sampling {sample}")
    outpath = FLUX_SAMPLES_PATH / f"{sample}.parquet"
    if outpath.exists():
        continue
    model = metworkpy.read_model(sample_path)
    logger.info("Starting OptGP Sampler")
    sampler = OptGPSampler(model=model, thinning=THINNING, processes=PROCESSES)
    samples = sampler.sample(NUM_SAMPLES)
    valid_samples = samples[sampler.validate(samples) == "v"]
    logger.info(f"Validated samples, {len(valid_samples)/len(samples):.2%} valid")
    logger.info("Saving Samples")
    valid_samples.to_parquet(outpath, index=False)
logger.info("Finished! ;-)")
