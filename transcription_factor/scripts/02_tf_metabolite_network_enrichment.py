"""
Script to determine the enrichment of TF targets in every metabolite network
"""

# Setup
# Imports
# Standard Library Imports
import logging
import pathlib

# External Imports
import metworkpy
import pandas as pd
from scipy import stats

# Local Imports

# Run Parameters
PROCESSES = 12

# Path Setup
BASE_PATH = pathlib.Path(__file__).parent.parent
DATA_PATH = BASE_PATH / "data"
MODDEL_PATH = BASE_PATH.parent / "models"
RESULTS_PATH = BASE_PATH / "results"

# Read in Base Model
BASE_MODEL = metworkpy.read_model(MODDEL_PATH / "iEK1011_m7H10_media.json")
MODEL_GENE_SET = set(BASE_MODEL.genes.list_attr("id"))

# Setup Logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    filename=BASE_PATH / "diagnostics" / "02_tf_metabolite_network_enrichment.log",
    filemode="w",
    level=logging.INFO,
)

# Determine TF targets for each kinase
logger.info("Finding TF Targets")
# Read the Fold change data
tf_fold_change = pd.read_excel(
    DATA_PATH / "tfoe_targets.xlsx",
    sheet_name="SupplementaryTableS2",
    skiprows=[0, 1, 2, 3, 4, 5, 6, 7, 9],
    header=0,
    index_col=0,
    usecols="A,E:HB",
)
# read the p-value data
tf_pval = pd.read_excel(
    DATA_PATH / "tfoe_targets.xlsx",
    sheet_name="SupplementaryTableS2",
    skiprows=[0, 1, 2, 3, 4, 5, 6, 7, 9],
    header=0,
    index_col=0,
    usecols="A,HC:OZ",
)
# Remove the .1 in the column names caused by repitition of columns,
# despite those columns not being present in the final dataframe >;-(
tf_pval.columns = tf_pval.columns.str.replace(".1", "")
# Create a target dataframe, where the TF column targets row gene when True
tf_targets = (tf_fold_change.abs() >= 1.0) & (tf_pval < 0.01)

# Next determine the Metabolite Networks
logger.info("Finding Metabolite Networks")

metabolite_network_out_path = RESULTS_PATH / "metabolite_network.csv"
if metabolite_network_out_path.exists():
    metabolite_network = pd.read_csv(metabolite_network_out_path, index_col=0, header=0)
else:
    metabolite_network = (
        metworkpy.metabolites.metabolite_network.find_metabolite_network_genes(
            model=BASE_MODEL.copy(),
            method="essential",
            essential_proportion=0.05,
            progress_bar=False,
            processes=PROCESSES,
        )
    )
    metabolite_network.to_csv(metabolite_network_out_path, index=True)

# Find the genes common to both
common_genes = list(set(metabolite_network.index).intersection(tf_targets.index))
# Filter for only these genes
tf_targets = tf_targets.loc[common_genes]
metabolite_network = metabolite_network.loc[common_genes]

# Create the results dataframe
enrichment_df = pd.DataFrame(
    {
        "tf": pd.Series(dtype="string"),
        "metabolite": pd.Series(dtype="string"),
        "total_genes": pd.Series(dtype=int),
        "tf_targets": pd.Series(dtype=int),
        "metabolite_network_size": pd.Series(dtype=int),
        "overlap": pd.Series(dtype=int),
        "p-value": pd.Series(dtype="float"),
    }
)
metabolite_enrichment_df_list = list()
for tf in tf_targets.columns:
    logger.info(f"Starting transcription factor {tf}")
    res_df = pd.DataFrame(
        0.0,
        index=metabolite_network.columns,
        columns=pd.Index(
            [
                "total_genes",
                "tf_targets",
                "metabolite_network_size",
                "overlap",
                "p-value",
            ]
        ),
    )
    res_df["tf"] = tf
    res_df["total_genes"] = len(common_genes)
    res_df["tf_targets"] = tf_targets[tf].sum()
    for metabolite in res_df.index:
        logger.info(f"Starting metabolite {metabolite}")
        res_df.loc[metabolite, "metabolite_network_size"] = metabolite_network[
            metabolite
        ].sum()
        res_df.loc[metabolite, "overlap"] = (
            metabolite_network[metabolite] & tf_targets[tf]
        ).sum()
        res_df.loc[metabolite, "p-value"] = stats.hypergeom.sf(
            M=res_df.loc[metabolite, "total_genes"],
            m=res_df.loc[metabolite, "tf_targets"],
            N=res_df.loc[metabolite, "metabolite_network_size"],
            k=res_df.loc[metabolite, "overlap"],
        )
    res_df = res_df.reset_index(names="metabolite")[
        [
            "tf",
            "metabolite",
            "total_genes",
            "tf_targets",
            "metabolite_network_size",
            "overlap",
            "p-value",
        ]
    ]
logger.info("Combining Results")
metabolite_enrichment_df = pd.concat(metabolite_enrichment_df_list, axis=0)
logger.info("Adjusting p-values to account for False Discovery Rate")
metabolite_enrichment_df["adjusted p-value"] = stats.false_discovery_control(
    metabolite_enrichment_df["p-value"], axis=None, method="bh"
)

logger.info("Saving Results")
metabolite_enrichment_df.to_csv(
    RESULTS_PATH / "tf_metabolite_network_enrichment.csv", index=False
)
logger.info("Finished :-)")
