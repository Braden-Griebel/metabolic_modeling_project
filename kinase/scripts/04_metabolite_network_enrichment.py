"""
Find the Metabolite networks and determine if any of networks
are enriched for kinase targeted genes
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

# Path Setup
BASE_PATH = pathlib.Path(__file__).parent.parent
RESULTS_PATH = BASE_PATH / "results"
DATA_PATH = BASE_PATH / "data"
MODEL_PATH = BASE_PATH.parent / "models"

# Read Base Model
BASE_MODEL = metworkpy.read_model(MODEL_PATH / "iEK1011_m7H10_media.json")
MODEL_GENE_SET = set(BASE_MODEL.genes.list_attr("id"))

# Run Parameters
PROCESSES = 12

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    filename=BASE_PATH / "diagnostics" / "04_metabolite_network_enrichment.log",
    level=logging.INFO,
)

# Determine Kinase targets for each Kinase
logger.info("Finding Kinase Targets")
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
    df = pd.read_excel(DATA_PATH / "frando_phosphorylation.xlsx", sheet_name=kinase)
    df = df[df["Indirect?"] != "Indirect"]
    kinase_targets[kinase] = set(df["Rv Number"]).intersection(MODEL_GENE_SET)


# Identify Metabolite Networks
logger.info("Determining Metabolite Networks")
# NOTE: Index is genes, Columns are metabolites
metabolite_networks = metworkpy.metabolites.find_metabolite_network_genes(
    BASE_MODEL, method="essential", processes=PROCESSES, progress_bar=False
)
logger.info("Found metabolite networks, saving results")
metabolite_networks.to_csv(RESULTS_PATH / "metabolite_networks.csv", index=True)

# Find enrichment of targets in any of the metabolite networks
metabolite_enrichment_df_list = []
logger.info("Finding Kinase Target Enrichment")
for kinase, targets in kinase_targets.items():
    logger.info(f"Starting {kinase}")
    res_df = pd.DataFrame(
        0.0,
        index=metabolite_networks.columns,
        columns=pd.Index(
            [
                "total_genes",
                "kinase_targets",
                "metabolite_network_size",
                "overalap",
                "p-value",
            ]
        ),
    )
    res_df["kinase"] = kinase
    res_df["total_genes"] = len(MODEL_GENE_SET)
    res_df["kinase_targets"] = len(targets)
    for metabolite in res_df.index:
        logger.info(f"Starting metabolite {metabolite}")
        m_series = metabolite_networks[metabolite]  # pd.Series
        m_network = set(m_series[m_series].index)
        res_df.loc[metabolite, "metabolite_network_size"] = len(m_network)
        res_df.loc[metabolite, "overlap"] = len(targets.intersection(m_network))
        res_df.loc[metabolite, "p-value"] = stats.hypergeom.sf(
            M=res_df.loc[metabolite, "total_genes"],
            n=res_df.loc[metabolite, "kinase_targets"],
            N=res_df.loc[metabolite, "metabolite_network_size"],
            k=res_df.loc[metabolite, "overlap"],
        )
    res_df = res_df.reset_index(names=["metabolite"])[
        [
            "kinase",
            "metabolite",
            "total_genes",
            "kinase_targets",
            "metabolite_network_size",
            "overlap",
            "p-value",
        ]
    ]
    metabolite_enrichment_df_list.append(res_df)
logger.info("Combining Results")
metabolite_enrichment_df = pd.concat(
    metabolite_enrichment_df_list, axis=0, ignore_index=True
)
logger.info("Adjusting p-value to account for false discovery rate")
metabolite_enrichment_df["adjusted p-value"] = stats.false_discovery_control(
    metabolite_enrichment_df["p-value"], axis=None, method="bh"
)

# Save the results
logger.info("Saving enrichment results")
metabolite_enrichment_df.to_csv(RESULTS_PATH / "metabolite_network_enrichment.csv")
