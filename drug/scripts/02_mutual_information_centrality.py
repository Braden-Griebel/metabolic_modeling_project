"""
Find mutual information networks, and associated centrality measures for all drug models
"""

# Setup
# Imports
# Standard Library Imports
import logging
import pathlib
from typing import cast

# External Imports
import cobra
import metworkpy
import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse

# Local Imports

# Path Setup
BASE_PATH = pathlib.Path(__file__).parent.parent
FLUX_SAMPLES_PATH = BASE_PATH / "flux_samples"
RESULTS_PATH = BASE_PATH / "results"
ADJACENCY_MATRIX_PATH = RESULTS_PATH / "mi_network_adj"
MODEL_PATH = BASE_PATH.parent / "models"

# Run Parameters
PROCESSES = 12
N_NEIGHBORS = 5
ZERO_DELTA = 1e-30  # Cutoff for value to be considered 0

# Determine the min and max using FVA to scale
BASE_MODEL = metworkpy.read_model(MODEL_PATH / "iEK1011_m7H10_media.json")
fva = cobra.flux_analysis.flux_variability_analysis(
    BASE_MODEL, loopless=False, fraction_of_optimum=0.95, processes=PROCESSES
)


def fva_scale(samples_df: pd.DataFrame, fva_df: pd.DataFrame):
    """Scale the samples dataframe using the fva results"""
    scaled_df = samples_df.sub(fva_df["minimum"], axis="columns").div(
        (fva_df["maximum"] - fva_df["minimum"]), axis="columns"
    )
    return scaled_df


# Setup Logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    filename=BASE_PATH / "diagnostics" / "02_mutual_information_centrality.log",
    filemode="w",
    level=logging.INFO,
)

logger.info("Starting to find MI networks...")
# Start by calculating the MI adjacency matrixed for all of the imat models
for sample_path in FLUX_SAMPLES_PATH.glob("*.parquet"):
    sample_name = sample_path.name.split(".")[0]
    logger.info(f"Starting with sample {sample_name}")
    mi_adj_outpath = ADJACENCY_MATRIX_PATH / f"{sample_name}.npz"
    # Check if the array has already been created
    if mi_adj_outpath.exists():
        continue
    # Read in flux sample
    flux_samples = pd.read_parquet(sample_path)
    num_rxns = len(flux_samples.columns)
    # Find all the indices of all 0 columns
    full_index = flux_samples.columns
    max_min_diff = flux_samples.max(axis=0) - flux_samples.min(axis=0)

    # Scale the flux samples to be 0-1
    scaled_samples = fva_scale(flux_samples, fva)
    scaled_samples = (
        scaled_samples.replace(np.inf, np.nan)
        .replace(-np.inf, np.nan)
        .dropna(axis="columns")
    )
    # Filtering for only columns which are not all 0
    non_zero_cols = max_min_diff[
        (max_min_diff > ZERO_DELTA) & (max_min_diff.index.isin(scaled_samples.columns))
    ].index
    scaled_samples = scaled_samples[non_zero_cols]
    # Calculate the MI for every rxn pair
    mi_arr = (
        metworkpy.information.mutual_information_network.mi_network_adjacency_matrix(
            scaled_samples,
            n_neighbors=N_NEIGHBORS,
            processes=PROCESSES,
            progress_bar=False,
        )
    )
    np.fill_diagonal(mi_arr, 0.0)

    # Create a sparse array to hold full adjacency matrix
    full_mi_arr = sparse.dok_array((num_rxns, num_rxns), np.float64)

    # Set the non-zero values
    full_mi_arr[
        full_index.get_indexer(non_zero_cols).reshape(-1, 1),
        full_index.get_indexer(non_zero_cols),
    ] = mi_arr

    # Remove negative values
    full_mi_arr = full_mi_arr.tocsc()
    full_mi_arr.data[full_mi_arr.data < 0.0] = 0.0
    full_mi_arr.eliminate_zeros()

    # Save the MI Adjacency Array
    logger.info("Saving the mutual information network as a sparse adjacency array")
    sparse.save_npz(mi_adj_outpath, full_mi_arr)


# Next calculate all the centrality measures for the mutual information network
# Detemrine if a results dataframe has already been created, if it has read it in
logger.info("Reading in or creating centrality results dataframe")
centrality_df_path = RESULTS_PATH / "information_centrality.csv"
if centrality_df_path.exists():
    centrality_df = pd.read_csv(centrality_df_path)
else:
    centrality_df = pd.DataFrame(
        {
            "sample": pd.Series(dtype="string"),
            "reaction": pd.Series(dtype="string"),
            "mean_degree_centrality": pd.Series(dtype="float"),
            "pagerank": pd.Series(dtype="float"),
            "eigenvalue": pd.Series(dtype="float"),
            "betweeness": pd.Series(dtype="float"),
            "closeness": pd.Series(dtype="float"),
        }
    )
RENAME_DICT = {idx: rxn.id for idx, rxn in enumerate(BASE_MODEL.reactions)}
REACTION_LIST = BASE_MODEL.reactions.list_attr("id")
for network_adj_path in ADJACENCY_MATRIX_PATH.glob("*.npz"):
    sample_name = network_adj_path.name.split(".")[0]
    logger.info(f"Starting to compute centrality for: {sample_name}")
    # Check if the sample has already been processed
    if len(centrality_df[centrality_df["sample"] == sample_name]) > 0:
        continue
    # Read in the adjacency matrix
    adj_arr = sparse.load_npz(network_adj_path)
    # Create a temporary results dataframe to append to the centrality_df
    res_df = pd.DataFrame(
        0.0,
        columns=pd.Index(
            [
                "mean_degree_centrality",
                "pagerank",
                "eigenvalue",
                "betweeness",
                "closeness",
            ]
        ),
        index=pd.Index(REACTION_LIST),
    )
    # Calculate the cenrality values
    # First the mean degree centrality since that will be based on the adj_arr only
    logger.info("Calculating the mean degree centrality")
    mean_degree = (adj_arr.sum(axis=0) / 2) / len(REACTION_LIST)
    mean_degree = pd.Series(mean_degree, index=pd.Index(REACTION_LIST))
    res_df.loc[mean_degree.index, "mean_degree_centrality"] = mean_degree

    # Calculate the Pagerank, and Eigenvalue centrality
    # Construct the network from the adjacency matrix
    mi_network = nx.from_scipy_sparse_array(adj_arr, create_using=nx.Graph)
    mi_network = cast(nx.Graph, mi_network)
    # Rename the nodes to reaction names
    mi_network = nx.relabel_nodes(mi_network, RENAME_DICT)
    # Calculate the pagerank centrality
    logger.info("Calculating pagerank centrality")
    pg_cent = pd.Series(nx.pagerank(mi_network, weight="weight"))
    # Calculate the eigenvalue centrality
    logger.info("Calculating eigenvalue centrality")
    eig_cent = pd.Series(nx.eigenvector_centrality(mi_network, weight="weight"))
    # Save the values to the dataframe
    res_df.loc[pg_cent.index, "pagerank"] = pg_cent
    res_df.loc[eig_cent.index, "eigenvalue"] = eig_cent

    # Calculate the betweeness and closeness
    # Note, that these treat weight as distance
    # so the reciprocal is taken for the non-zero entries in the adjacency matrix
    logger.info("Calculating reciprocal of the adjacency matrix")
    adj_arr_reciprocal = adj_arr
    adj_arr_reciprocal.data[adj_arr_reciprocal.data > 0.0] = np.reciprocal(
        adj_arr_reciprocal.data[adj_arr_reciprocal.data > 0.0]
    )
    # Create the network
    mi_network_reciprocal = nx.from_scipy_sparse_array(
        adj_arr_reciprocal, create_using=nx.Graph
    )
    # Relabel the nodes
    mi_network_reciprocal = nx.relabel_nodes(mi_network_reciprocal, RENAME_DICT)
    mi_network_reciprocal = cast(nx.Graph, mi_network_reciprocal)
    # Calculate the centrality
    logger.info("Calculating the betweeness centrality")
    betweeness_centrality = pd.Series(
        nx.betweenness_centrality(mi_network_reciprocal, weight="weight")
    )
    logger.info("Calculating closeness centrality")
    closeness_centrality = pd.Series(
        nx.closeness_centrality(mi_network_reciprocal, distance="weight")
    )
    # Save the values to the dataframe
    res_df.loc[betweeness_centrality.index, "betweeness"] = betweeness_centrality
    res_df.loc[closeness_centrality.index, "closeness"] = closeness_centrality

    # Add the sample name to the res_df and append it to the centrality_df
    res_df["sample"] = sample_name
    res_df = res_df.reset_index(names="reaction")
    res_df = res_df[
        [
            "sample",
            "reaction",
            "mean_degree_centrality",
            "pagerank",
            "eigenvalue",
            "betweeness",
            "closeness",
        ]
    ]
    logger.info(
        "Adding the results for this sample to the overall centrality dataframe"
    )
    centrality_df = pd.concat([centrality_df, res_df], axis=0)
    # Save the intermediate results
    logger.info("Saving intermediate results")
    centrality_df.to_csv(centrality_df_path, index=False)
# Save the final results
logger.info("Saving final results")
centrality_df.to_csv(centrality_df_path, index=False)
logger.info("Done :-)")
