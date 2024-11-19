"""
Script to generate IMAT models for each Kinase gene expression sample, and sample from them.
"""

# SETUP
# Imports
# Standard Library Imports
import itertools
import logging
import pathlib
import warnings

# External Imports
from cobra.sampling import OptGPSampler
import metworkpy
import pandas as pd

# Local Imports

# Setup Path
BASE_PATH = pathlib.Path(__file__).parent.parent
DATA_PATH = BASE_PATH / "data"
RESULTS_PATH = BASE_PATH / "results"
FLUX_SAMPLES_PATH = BASE_PATH / "flux_samples"
MODEL_PATH = BASE_PATH.parent / "models"
MODEL_OUT_PATH = BASE_PATH / "kinase_models"


# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    filename=BASE_PATH / "diagnostics" / "01_model_generation_sampling.log",
    filemode="w",
    level=logging.INFO,
)

# Read in base Model
logger.info("Reading in m7H10 model")
BASE_MODEL = metworkpy.read_model(MODEL_PATH / "iEK1011_m7H10_media.json")

# Run Parameters
PROCESSES = 12
QUANTILE = (0.15, 0.85)
EPSILON = 1.0
THRESHOLD = 0.01
GENE_SUBSET = BASE_MODEL.genes.list_attr("id")
THINNING = 150
NUM_SAMPLES = 1_000

# Read in the Expression Data
logger.info("Reading in expression data")
expr_rpkm = pd.read_csv(
    DATA_PATH / "frando_gene_expr_rpkm.csv",
    index_col=0,
    usecols=lambda x: x not in ["GrowthPhase"],
).drop(["WT10", "WT11"])
# NOTE: WT10 and WT11 dropped because there was apparently some issues with those samples


# Seperate WT and Sample gene expression
wt_gene_expr = expr_rpkm[expr_rpkm.index.str.startswith("WT")]
sample_gene_expr = expr_rpkm[~expr_rpkm.index.str.startswith("WT")]

for sample in expr_rpkm.index:
    logger.info(f"Starting {sample}")
    model_out_path = MODEL_OUT_PATH / f"{sample}.json"
    if model_out_path.exists():
        # Model has already been created so skip it
        continue
    gene_weights_df_path = RESULTS_PATH / "imat_info" / "gene_weights.csv"
    gene_weights_df = pd.DataFrame(
        0.0, index=pd.Index(GENE_SUBSET), columns=pd.Index(["weight"])
    )
    gene_weights_df["sample"] = sample
    expr_series = expr_rpkm.loc[sample]
    # Convert gene expression to gene weights
    logger.info("Converting gene expression to gene weights")
    gene_weights = metworkpy.utils.expr_to_imat_gene_weights(
        expression=expr_series,
        quantile=QUANTILE,
        subset=GENE_SUBSET,
    )
    gene_weights_df.loc[gene_weights.index, "weight"] = gene_weights
    gene_weights_df = gene_weights_df.reset_index(names="gene")
    gene_weights_df = gene_weights_df[["sample", "gene", "weight"]]
    # Save gene weights, append if the file already exists
    if gene_weights_df_path.exists():
        df = pd.read_csv(gene_weights_df_path)
        if len(df[df["sample"] == sample]) == 0:
            gene_weights_df.to_csv(
                gene_weights_df_path, index=False, header=False, mode="a"
            )
    else:
        gene_weights_df.to_csv(gene_weights_df_path, index=False, header=True, mode="w")
    # Convert gene weights to reaction weights
    logger.info("Converting gene weights to reaction weights")
    rxn_weights_df_path = RESULTS_PATH / "imat_info" / "reaction_weights.csv"
    rxn_weights_df = pd.DataFrame(
        0.0,
        index=pd.Index(BASE_MODEL.reactions.list_attr("id")),
        columns=pd.Index(["weight"]),
    )
    rxn_weights_df["sample"] = sample
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
    rxn_weights_df.loc[rxn_weights.index, "weight"] = rxn_weights
    rxn_weights_df = rxn_weights_df.reset_index(names="reaction")
    rxn_weights_df = rxn_weights_df[["sample", "reaction", "weight"]]
    # Save the reaction weights, append if the file already exists
    if rxn_weights_df_path.exists():
        df = pd.read_csv(rxn_weights_df_path)
        if len(df[df["sample"] == sample]) == 0:
            rxn_weights_df.to_csv(
                rxn_weights_df_path, index=False, header=False, mode="a"
            )
    else:
        rxn_weights_df.to_csv(rxn_weights_df_path, index=False, header=True, mode="w")
    # Generate IMAT Model
    logger.info("Generating FVA Model")
    imat_model = metworkpy.imat.generate_model(
        model=BASE_MODEL,
        rxn_weights=rxn_weights,
        method="fva",
        epsilon=EPSILON,
        threshold=THRESHOLD,
        loopless=False,  # Issues with loopless solution
        processes=PROCESSES,
    )
    # Write model to file
    logger.info("Saving IMAT Model")
    metworkpy.write_model(model=imat_model, model_path=model_out_path)

# Finally, generate a combined WT Model
logger.info("Generating Combined WT Model")
wt_model_out_path = MODEL_OUT_PATH / "WT_combined.json"
if not wt_model_out_path.exists():
    wt_gene_weights_df_path = RESULTS_PATH / "imat_info" / "gene_weights.csv"
    wt_gene_weights_df = pd.DataFrame(
        0.0, index=pd.Index(GENE_SUBSET), columns=pd.Index(["weight"])
    )
    wt_gene_weights_df["sample"] = "WT_combined"
    wt_gene_expr_series = wt_gene_expr.median(axis=0)
    logger.info("Finding WT gene weights")
    wt_gene_weights = metworkpy.utils.expr_to_imat_gene_weights(
        expression=wt_gene_expr_series, quantile=QUANTILE, subset=GENE_SUBSET
    )
    wt_gene_weights_df.loc[wt_gene_weights.index, "weight"] = wt_gene_weights
    wt_gene_weights_df = wt_gene_weights_df.reset_index(names="gene")
    wt_gene_weights_df = wt_gene_weights_df[["sample", "gene", "weight"]]
    if wt_gene_weights_df_path.exists():
        df = pd.read_csv(wt_gene_weights_df_path)
        if len(df[df["sample"] == "WT_combined"]) == 0:
            wt_gene_weights_df.to_csv(
                wt_gene_weights_df_path, index=False, header=False, mode="a"
            )
    else:
        wt_gene_weights_df.to_csv(
            wt_gene_weights_df_path, index=False, header=True, mode="w"
        )
    logger.info("Finding WT reaction weights")
    wt_rxn_weights_df_path = RESULTS_PATH / "imat_info" / "reaction_weights.csv"
    wt_rxn_weights_df = pd.DataFrame(
        0.0,
        index=pd.Index(BASE_MODEL.reactions.list_attr("id")),
        columns=pd.Index(["weight"]),
    )
    wt_rxn_weights_df["sample"] = "WT_combined"
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Genes .* are in model but not in gene weights, setting their weights to .*",
            category=UserWarning,
            module=r".*metworkpy.*",
        )
        wt_rxn_weights = metworkpy.gpr.gene_to_rxn_weights(
            model=BASE_MODEL,
            gene_weights=wt_gene_weights,
            fn_dict=metworkpy.gpr.gpr_functions.IMAT_FUNC_DICT,
            fill_val=0,
        )
    wt_rxn_weights_df.loc[wt_rxn_weights.index, "weight"] = wt_rxn_weights
    wt_rxn_weights_df = wt_rxn_weights_df.reset_index(names="reaction")
    wt_rxn_weights_df = wt_rxn_weights_df[["sample", "reaction", "weight"]]
    if wt_rxn_weights_df_path.exists():
        df = pd.read_csv(wt_rxn_weights_df_path)
        if len(df[df["sample"] == "WT_combined"]) == 0:
            wt_rxn_weights_df.to_csv(
                wt_rxn_weights_df_path, index=False, header=False, mode="a"
            )
    else:
        wt_rxn_weights_df.to_csv(
            wt_rxn_weights_df_path, index=False, header=True, mode="w"
        )
    logger.info("Generating WT model")
    wt_imat_model = metworkpy.imat.generate_model(
        model=BASE_MODEL,
        rxn_weights=wt_rxn_weights,
        method="fva",
        epsilon=EPSILON,
        threshold=THRESHOLD,
        loopless=False,
        processes=PROCESSES,
    )
    logger.info("Saving Model To File")
    metworkpy.write_model(model=wt_imat_model, model_path=wt_model_out_path)

logger.info("Finding reaction bounds for all models and Saving")
rxn_bounds_df_list = []
for sample in itertools.chain(expr_rpkm.index, ["WT_combined", "base_model"]):
    if sample == "base_model":
        model = BASE_MODEL
    else:
        model = metworkpy.read_model(MODEL_OUT_PATH / f"{sample}.json")
    bounds_df = pd.DataFrame(
        0.0,
        index=pd.Index(model.reactions.list_attr("id")),
        columns=pd.Index(["lb", "ub"]),
    )
    bounds_df["sample"] = sample
    bounds_df["lb"] = model.reactions.list_attr("lower_bound")
    bounds_df["ub"] = model.reactions.list_attr("upper_bound")
    bounds_df = bounds_df.reset_index(names="reaction")
    bounds_df = bounds_df[["sample", "reaction", "lb", "ub"]]
    rxn_bounds_df_list.append(bounds_df)
# Concat all the bounds_df
rxn_bounds_df = pd.concat(rxn_bounds_df_list, axis=0, ignore_index=True)
rxn_bounds_df.to_csv(
    RESULTS_PATH / "imat_info" / "reaction_bounds.csv", index=False, header=True
)

# Generate Samples for all the IMAT Models
logger.info("Starting flux sampling")
for sample in itertools.chain(expr_rpkm.index, ["WT_combined"]):
    logger.info(f"Sampling from {sample}")
    flux_sample_path = FLUX_SAMPLES_PATH / f"{sample}.parquet"
    if flux_sample_path.exists():
        continue
    model = metworkpy.read_model(MODEL_OUT_PATH / f"{sample}.json")
    logger.info("Starting OptGP Sampler")
    sampler = OptGPSampler(model=model, thinning=THINNING, processes=PROCESSES)
    samples = sampler.sample(NUM_SAMPLES)
    valid_samples = samples[sampler.validate(samples) == "v"]
    logger.info(f"Validated samples, {len(valid_samples)/NUM_SAMPLES:.2%} valid")
    logger.info("Saving Samples")
    valid_samples.to_parquet(
        flux_sample_path,
        index=False,
    )
