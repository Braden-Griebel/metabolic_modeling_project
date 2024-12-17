"""
Script for generating IMAT models for all drug conditions and performing flux sampling
"""

# Setup
# Imports
# Standard Library Imports
import logging
import pathlib
from typing import cast
import re
import warnings

# External Imports
import cobra.exceptions
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
MODEL_OUT_PATH = BASE_PATH / "imat_models"

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    filename=BASE_PATH / "diagnostics" / "01_model_generation_sampling.log",
    filemode="w",
    level=logging.INFO,
)

# Read in the Base model
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
OBJECTIVE_TOLERANCE = 5e-2


# Read in the expression data
logger.info("Reading in expression data")
expr = (
    pd.read_csv(DATA_PATH / "INDIGO-transcriptomes-all_v1.csv", index_col=0)
    .reset_index(names=["sample"])
    .set_index("sample")
)  # bad way to change name
# read in the metadata
expr_meta = pd.read_csv(DATA_PATH / "INDIGO-metadata.csv", index_col=0).reset_index(
    names=["sample"]
)
expr_meta = cast(pd.DataFrame, expr_meta)
# Read in the mycobrowser data to find gene lengths
myco_df = pd.read_csv(
    DATA_PATH / "mycobrowser" / "Mycobacterium_tuberculosis_H37Rv_txt_v5.txt", sep="\t"
)
myco_df["length"] = (myco_df["Stop"] - myco_df["Start"]).abs()
myco_df = myco_df[myco_df["Feature"] == "CDS"]
lengths = myco_df.set_index("Locus")["length"]
lengths = cast(pd.Series, lengths)
# Free the myco_df since it won't be needed anymore
del myco_df

logger.info("Converting expr data to rpkm")
expr_rpkm = metworkpy.utils.count_to_rpkm(
    expr.loc[:, "Rv0001":], feature_length=lengths
)


logging.info("Beggining to generate models")
# Start with non-reference samples, which will all have models generated
for sample in expr_meta[expr_meta["Reference"] == 0]["sample"]:
    logger.info(f"Starting sample: {sample}")
    out_path = MODEL_OUT_PATH / f"{sample}.json"
    if out_path.exists():
        continue  # already generated model for this
    # determine the gene weights
    try:
        expr_series = expr_rpkm.loc[sample]
    except KeyError as _:
        logging.exception("Sample not in expression data")
        continue
    assert isinstance(expr_series, pd.Series)  # Make sure that it returns a series
    gene_weights = metworkpy.utils.expr_to_imat_gene_weights(
        expression=expr_series, quantile=QUANTILE, subset=GENE_SUBSET
    )
    logger.info(f"Found {(gene_weights==1.).sum()} upregulated genes")
    logger.info(f"Found {(gene_weights==-1.).sum()} downregulated genes")
    logger.info(f"Found {(gene_weights==0.).sum()} nonregulated genes")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        rxn_weights = metworkpy.gpr.gene_to_rxn_weights(
            BASE_MODEL,
            gene_weights=gene_weights,
            fn_dict=metworkpy.gpr.gpr_functions.IMAT_FUNC_DICT,
        )
    logger.info(f"Found {(rxn_weights==1.).sum()} upregulated reactions")
    logger.info(f"Found {(rxn_weights==-1.).sum()} downregulated reactions")
    logger.info(f"Found {(rxn_weights==0.).sum()} nonregulated reactions")
    logger.info("Finding IMAT Model")
    try:
        out_model = metworkpy.imat.model_creation.generate_model(
            model=BASE_MODEL,
            rxn_weights=rxn_weights,
            method="fva",
            loopless=False,
            epsilon=EPSILON,
            threshold=THRESHOLD,
            objective_tolerance=OBJECTIVE_TOLERANCE,
            processes=PROCESSES,
        )
    except (
        cobra.exceptions.Infeasible,
        cobra.exceptions.OptimizationError,
        cobra.exceptions.Unbounded,
        cobra.exceptions.FeasibleButNotOptimal,
        cobra.exceptions.UndefinedSolution,
    ) as _:
        logger.exception(f"Error finding model for sample: {sample}")
        continue
    logger.info("Writing resulting file to output")
    metworkpy.utils.write_model(out_model, out_path)

# Now work through the reference conditions
for (group, drug, dose, time), df in expr_meta[expr_meta["Reference"] == 1].groupby(
    by=["Group", "Drug", "Dose", "Timept_h"]
):
    logger.info(f"Starting group: {group}, drug: {drug}, dose: {dose}, time: {time}")
    if group == -1:
        continue  # Skipping bad group
    # fix dose so is can be used as part of a path without /
    dose = re.sub("/", "_per_", dose)
    out_path = MODEL_OUT_PATH / f"g{group}_{drug}_{dose}_{time}_reference_fva_imat.json"
    if out_path.exists():
        continue  # Already completed this model
    # Determine the reaction weights
    gene_expr_df = expr_rpkm.loc[expr_rpkm.index.isin(df["sample"])]
    if len(gene_expr_df) == 0:
        logger.info("Sample not present in gene expression compendia")
        continue
    gene_weights = metworkpy.utils.expr_to_imat_gene_weights(
        expression=gene_expr_df, quantile=QUANTILE, sample_axis=0, subset=GENE_SUBSET
    )
    logger.info(f"Found {(gene_weights==1.).sum()} upregulated genes")
    logger.info(f"Found {(gene_weights==-1.).sum()} downregulated genes")
    logger.info(f"Found {(gene_weights==0.).sum()} nonregulated genes")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        rxn_weights = metworkpy.gpr.gene_to_rxn_weights(
            BASE_MODEL,
            gene_weights=gene_weights,
            fn_dict=metworkpy.gpr.gpr_functions.IMAT_FUNC_DICT,
        )
    logger.info(f"Found {(rxn_weights==1.).sum()} upregulated reactions")
    logger.info(f"Found {(rxn_weights==-1.).sum()} downregulated reactions")
    logger.info(f"Found {(rxn_weights==0.).sum()} nonregulated reactions")
    logger.info("Finding IMAT Model")
    try:
        out_model = metworkpy.imat.model_creation.generate_model(
            model=BASE_MODEL,
            rxn_weights=rxn_weights,
            method="fva",
            epsilon=EPSILON,
            threshold=THRESHOLD,
            objective_tolerance=OBJECTIVE_TOLERANCE,
            loopless=False,
            processes=PROCESSES,
        )
    except (
        cobra.exceptions.Infeasible,
        cobra.exceptions.OptimizationError,
        cobra.exceptions.Unbounded,
        cobra.exceptions.FeasibleButNotOptimal,
        cobra.exceptions.UndefinedSolution,
    ):
        logger.exception(f"Error finding model for group {group} reference")
        continue
    logger.info("Writing results file to output")
    metworkpy.utils.write_model(out_model, out_path)

logger.info("Beggining to sample from the generated models")
for path in MODEL_OUT_PATH.glob("*"):
    name = path.name.split(".")[0]
    logger.info(f"Starting to sample from {name}")
    out_path = FLUX_SAMPLES_PATH / f"{name}.parquet"
    if out_path.exists():
        continue  # samples already generated
    model = metworkpy.read_model(path)
    sampler = OptGPSampler(
        thinning=500,
        processes=PROCESSES,
        model=model,
    )
    samples = sampler.sample(NUM_SAMPLES)
    valid_samples = samples[sampler.validate(samples) == "v"]
    logger.info(f"{len(samples)/len(valid_samples):%} of samples were valid")
    logger.info("writing samples to output")
    samples.to_parquet(out_path, index=False)
logger.info("Finally finished Hooray!:-)")
