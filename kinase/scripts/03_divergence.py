"""
Calculate the divergence for each reaction  between the kinase samples and the WT samples
"""

# Setup
# Imports
# Standard Library Imports
import logging
import pathlib
from typing import cast
import warnings

# External Imports
import cobra
import metworkpy
from metworkpy.divergence.kl_divergence_functions import kl_divergence_array
import numpy as np
import pandas as pd

# Local Imports

# Ignore some warnings that show up
warnings.filterwarnings(
    "ignore",
    message=r".*invalid value encountered in divide.*",
    category=RuntimeWarning,
    module=r".*divergence.*",
)
warnings.filterwarnings(
    "ignore",
    message=r".*divide by zero encountered in divide.*",
    category=RuntimeWarning,
    module=r".*divergence.*",
)

# Path Setup
BASE_PATH = pathlib.Path(__file__).parent.parent
DATA_PATH = BASE_PATH / "data"
RESULTS_PATH = BASE_PATH / "results"
FLUX_SAMPLES_PATH = BASE_PATH / "flux_samples"

# NOTE: Two methods for generating WT sample for comparison, can use WT_combined, or concat WT1-9

# Run Parameters
PROCESSES = 12
FRACTION_OF_OPTIMUM = 0.95
N_NEIGHBORS = 5
METRIC = 2.0


# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    filename=BASE_PATH / "diagnostics" / "03_divergence.log",
    filemode="w",
    level=logging.INFO,
)

# Read in gene expression to get a list of samples
logger.info("Reading in expression data")
expr_rpkm = pd.read_csv(
    DATA_PATH / "frando_gene_expr_rpkm.csv",
    index_col=0,
    usecols=lambda x: x not in ["GrowthPhase"],
).drop(["WT10", "WT11"])

# Read in base model for reaction info
BASE_MODEL = metworkpy.read_model(
    BASE_PATH.parent / "models" / "iEK1011_m7H10_media.json"
)

# Use FVA to find min and max for unconstrained model for normalizing ditributions
fva_res = cobra.flux_analysis.variability.flux_variability_analysis(
    model=BASE_MODEL, loopless=False, fraction_of_optimum=FRACTION_OF_OPTIMUM
)
fva_res["diff"] = fva_res["maximum"] - fva_res["minimum"]


# FIRST METHOD: WT_combined
logger.info("Starting WT_combined divergence")
divergence_df_index = expr_rpkm.loc[~expr_rpkm.index.str.startswith("WT")].index
divergence_df = pd.DataFrame(
    0.0,
    index=divergence_df_index,
    columns=pd.Index(BASE_MODEL.reactions.list_attr("id")),
)


# Read in WT_combined samples
logger.info("Read in the WT_combined samples")
wt_combined = pd.read_parquet(FLUX_SAMPLES_PATH / "WT_combined.parquet")

# Scale wt_combined
logger.info("Scaling the WT_combined samples")
wt_combined = wt_combined.sub(fva_res["minimum"], axis="columns").div(
    fva_res["diff"], axis="columns"
)
# Drop columns where this results in np.nan
wt_combined = wt_combined.replace(np.inf, np.nan).dropna(axis="columns")

for sample in expr_rpkm.loc[~expr_rpkm.index.str.startswith("WT")].index:
    logger.info(f"Starting sample {sample}")
    flux_sample = pd.read_parquet(FLUX_SAMPLES_PATH / f"{sample}.parquet")
    # Standardize the flux samples
    logger.info("Standardizing the flux samples")
    flux_sample = flux_sample.sub(fva_res["minimum"], axis="columns").div(
        fva_res["diff"], axis="columns"
    )
    # Drop columns where this results in np.nan
    logger.info("Drop columns with np.nan")
    flux_sample = flux_sample.replace(np.inf, np.nan).dropna(axis="columns")
    # Find columns common to both
    logger.info("Finding common columns")
    common_columns = [
        r
        for r in wt_combined.columns
        if r in set(wt_combined.columns).intersection(flux_sample.columns)
    ]
    wt_common = wt_combined[common_columns]
    flux_sample = flux_sample[common_columns]
    # These will be dataframes, cast to eliminate type errors
    wt_common = cast(pd.DataFrame, wt_common)
    flux_sample = cast(pd.DataFrame, wt_common)
    # Compute the divergence for all of the columns
    logger.info("Finding divergence")
    div_res = kl_divergence_array(
        p=wt_common,
        q=flux_sample,
        n_neighbors=N_NEIGHBORS,
        metric=METRIC,
        processes=PROCESSES,
    )
    div_res = cast(pd.Series, div_res)  # Should be a series, indexed by reaction id
    divergence_df.loc[sample, div_res.index] = div_res
# Save the results of the WT_combined divergence data
logger.info("Saving Results")
divergence_df.to_csv(RESULTS_PATH / "divergence" / "divergence_wt_combined.csv")

# SECOND METHOD: Combine samples from Individual WT Models
logger.info("Starting finding divergence with combined WT1-9")
# Combine all the WT samples into a single dataframe
logger.info("Reading in WT samples")
wt_samples_list = []
for sample in expr_rpkm[expr_rpkm.index.str.startswith("WT")].index:
    wt_samples_list.append(pd.read_parquet(FLUX_SAMPLES_PATH / f"{sample}.parquet"))
wt_samples = pd.concat(wt_samples_list, axis=0, ignore_index=True)
# Down sample to 1000 samples (not strictly necessary, but more comparable)
logger.info("Down sampling")
wt_samples = wt_samples.sample(1000, axis="index")

divergence_df = pd.DataFrame(
    0.0,
    index=divergence_df_index,
    columns=pd.Index(BASE_MODEL.reactions.list_attr("id")),
)
# Standardize the WT samples
logger.info("Scaling the WT Samples")
wt_samples = wt_samples.sub(fva_res["minimum"], axis="columns").div(
    fva_res["diff"], axis="columns"
)
logger.info("Dropping missing data")
wt_samples = wt_samples.replace(np.inf, np.nan).dropna(axis="columns")


for sample in expr_rpkm.loc[~expr_rpkm.index.str.startswith("WT")].index:
    logger.info(f"Starting sample {sample}")
    flux_sample = pd.read_parquet(FLUX_SAMPLES_PATH / f"{sample}.parquet")
    # Standardize the flux samples
    logger.info("Scaling the flux samples")
    flux_sample = (
        flux_sample.sub(fva_res["minimum"], axis="columns")
        .div(fva_res["diff"], axis="columns")
        .replace(np.inf, np.nan)
        .dropna(axis="columns")
    )
    # Find common columns
    common_columns = [
        r
        for r in wt_samples.columns
        if r in set(wt_samples.columns).intersection(flux_sample.columns)
    ]
    wt_common = wt_samples[common_columns]
    flux_sample = flux_sample[common_columns]
    # Case to eliminate type errors
    wt_common = cast(pd.DataFrame, wt_common)
    flux_sample = cast(pd.DataFrame, flux_sample)
    logger.info("Finding divergence")
    div_res = kl_divergence_array(
        p=wt_common,
        q=flux_sample,
        n_neighbors=N_NEIGHBORS,
        metric=METRIC,
        processes=PROCESSES,
    )
    div_res = cast(pd.Series, div_res)
    divergence_df.loc[sample, div_res.index] = div_res
# Save the results
logger.info("Saving Divergence Results")
divergence_df.to_csv(RESULTS_PATH / "divergence" / "divergence_multiple_wt.csv")
