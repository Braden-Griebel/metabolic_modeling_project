"""
Generate data associated with the Model, such as mutual information centrality, ko-divergence, fva, etc.
"""

# Setup
# Imports
# Standard library imports
from collections import defaultdict
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
DATA_PATH = BASE_PATH / "data"
RESULTS_PATH = BASE_PATH / "results"


# Load Model
BASE_MODEL = metworkpy.read_model(
    BASE_PATH.parent / "models" / "iEK1011_m7H10_media.json"
)

# Run Parameters
PROCESSES = 12
THINNING = 500
NUM_SAMPLES = 1_000
ZERO_DELTA = 1e-30
N_NEIGHBORS = 5

# General Constants
REACTION_LIST = BASE_MODEL.reactions.list_attr("id")
GENE_LIST = BASE_MODEL.reactions.list_attr("id")


# Setup Logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    filename=BASE_PATH / "diagnostics" / "01_model_data_preparation.log",
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s [%(lineno)d]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Section: Setup Results DataFrame
rxn_results_df = pd.DataFrame(index=pd.Index(BASE_MODEL.reactions.list_attr("id")))
gene_results_df = pd.DataFrame(index=pd.Index(BASE_MODEL.genes.list_attr("id")))

# Section: FVA
logger.info("Performing FVA")
fva_res = cobra.flux_analysis.flux_variability_analysis(
    model=BASE_MODEL, loopless=True, fraction_of_optimum=0.95, processes=PROCESSES
)
rxn_results_df["fva_min"] = fva_res["minimum"]
rxn_results_df["fva_max"] = fva_res["maximum"]

# Section: Mutual Information Centrality
logger.info("Starting with Mutual Information Centrality")
flux_sample_path = BASE_PATH / "flux_samples" / "wt_sample.parquet"
if not flux_sample_path.exists():
    logger.info("Generating flux samples")
    sampler = cobra.sampling.OptGPSampler(
        model=BASE_MODEL, thinning=THINNING, processes=PROCESSES
    )
    flux_samples = sampler.sample(NUM_SAMPLES, fluxes=True)
    flux_samples = flux_samples[sampler.validate(flux_samples) == "v"]
    flux_samples.to_parquet(flux_sample_path, index=False)
else:
    logger.info("Reading in flux samples")
    flux_samples = pd.read_parquet(flux_sample_path)
# Contruct Mutual Information Network
logger.info("Constructing Mutual Information Network")
mi_adj_path = RESULTS_PATH / "mi_adjacency_matrix.npz"
if mi_adj_path.exists():
    logger.info("Reading in Adjacency Matrix")
    adj_arr = sparse.load_npz(mi_adj_path)
else:
    logger.info("Calculating Mutual Information Adjacency Matrix")
    # Start by filtering out columns that are all 0
    full_index = flux_samples.columns
    num_rxns = len(full_index)
    # Find the columns where all the samples are the same (within ZERO_DELTA)
    max_min_diff = flux_samples.max(axis=0) - flux_samples.min(axis=0)
    non_id_cols = max_min_diff[
        (max_min_diff > ZERO_DELTA) & (max_min_diff.index.isin(flux_samples.columns))
    ].index
    flux_samples = flux_samples[non_id_cols]
    flux_samples = cast(pd.DataFrame, flux_samples)
    # Calculate the MI for every rxn pair
    mi_arr = (
        metworkpy.information.mutual_information_network.mi_network_adjacency_matrix(
            samples=flux_samples,
            n_neighbors=N_NEIGHBORS,
            processes=PROCESSES,
            progress_bar=False,
        )
    )
    np.fill_diagonal(mi_arr, 0.0)

    # Create a sparse array to hold adjacency matrix
    full_mi_arr = sparse.dok_array((num_rxns, num_rxns), np.float64)

    # Set the non-zero values
    full_mi_arr[
        full_index.get_indexer(non_id_cols).reshape(-1, 1),
        full_index.get_indexer(non_id_cols),
    ] = mi_arr

    # Remove the negative values
    full_mi_arr = full_mi_arr.tocsc()
    full_mi_arr.data[full_mi_arr.data < 0.0] = 0.0
    full_mi_arr.eliminate_zeros()

    # Save the mi adjacency array
    logger.info("Saving adjacency array")
    sparse.save_npz(mi_adj_path, full_mi_arr)
    adj_arr = full_mi_arr
logger.info("Finding Mutual Information Centrality")
centrality_df = pd.DataFrame(
    0.0,
    index=pd.Index(BASE_MODEL.reactions.list_attr("id")),
    columns=pd.Index(
        ["pagerank", "eigenvector", "mean_degree", "closeness", "betweeness"]
    ),
)

RENAME_DICT = {idx: id for idx, id in enumerate(REACTION_LIST)}
# Start by calculating the mean degree centrality
logger.info("Calculating Mean Degree Centrality")
mean_degree = (adj_arr.sum(axis=0) / 2) / len(REACTION_LIST)
mean_degree = pd.Series(mean_degree, index=pd.Index(REACTION_LIST))
centrality_df.loc[mean_degree.index, "mean_degree"] = mean_degree

# Calculate the PageRank and Eigenvector centrality
logging.info("Finding the Pagerank centrality ")
mi_network = nx.from_scipy_sparse_array(adj_arr, create_using=nx.Graph)
mi_network = cast(nx.Graph, mi_network)
# Rename the nodes to match reaction name
mi_network = nx.relabel_nodes(mi_network, RENAME_DICT)
# Calulate the pagerank centrality
pg_cent = pd.Series(nx.pagerank(mi_network, weight="weight"))
# Calculate the eigenvalue centrality
logger.info("Finding the eigenvalue centrality")
eig_cent = pd.Series(nx.eigenvector_centrality(mi_network, weight="weight"))
centrality_df.loc[pg_cent.index, "pagerank"] = pg_cent
centrality_df.loc[eig_cent.index, "eigenvector"] = eig_cent


# Find the closeness and betweeness
logger.info("Finding reciprocal of mutual information")
adj_arr_recip = adj_arr
adj_arr_recip.data[adj_arr_recip.data > 0.0] = np.reciprocal(
    adj_arr.data[adj_arr_recip.data > 0.0]
)
mi_network_reciprocal = nx.from_scipy_sparse_array(adj_arr_recip, create_using=nx.Graph)
# Relabel the nodes
mi_network_reciprocal = nx.relabel_nodes(mi_network_reciprocal, RENAME_DICT)
logger.info("Finding betweeness centrality")
bet_cent = pd.Series(nx.betweenness_centrality(mi_network_reciprocal, weight="weight"))
logger.info("Finding closeness centrality")
clos_cent = pd.Series(nx.closeness_centrality(mi_network_reciprocal, distance="weight"))
logger.info("Saving results to dataframe")
centrality_df.loc[bet_cent.index, "betweeness"] = bet_cent
centrality_df.loc[clos_cent.index, "closeness"] = clos_cent
# Add the centrality data onto the rxn_results_df
rxn_results_df = pd.concat([rxn_results_df, centrality_df], axis=1, ignore_index=False)


# Section Metabolic Centrality
logger.info("Finding metabolic centrality")
logger.info("Generating Bipartite metabolic network")
metabolic_network = metworkpy.network.create_metabolic_network(
    BASE_MODEL, weighted=False, directed=False
)
logger.info("Projecting Bipartite network onto reactions only")
rxn_network = metworkpy.network.bipartite_project(
    metabolic_network, directed=False, weight=None, node_set=REACTION_LIST
)
logger.info("Finding reaction pagerank")
pg_cent = pd.Series(nx.pagerank(rxn_network))
pg_cent.name = "reaction_pagerank"

logger.info("Finding Eigenvector centrality")
eig_cent = pd.Series(nx.eigenvector_centrality(rxn_network))
eig_cent.name = "reaction_eigenvector_centrality"
logger.info("Finding Betweeness centrality")
bet_cent = pd.Series(nx.betweenness_centrality(rxn_network))
bet_cent.name = "reaction_betweeness_centrality"
logger.info("Finding closeness centrality")
clos_cent = pd.Series(nx.closeness_centrality(rxn_network))
clos_cent.name = "reaction_closeness_centrality"

rxn_results_df = pd.concat(
    [rxn_results_df, pg_cent, eig_cent, bet_cent, clos_cent], axis=1
)

# Section: ko-divergence
# First, Defining the different networks of interest,
# This will mainly just use the subsystems in the model, but will also include a whole metabolism one
logger.info("Finding KO divergence")
subsystem_dict = defaultdict(list)
logger.info("Finding subsystems")
for rxn in BASE_MODEL.reactions:
    subsys = rxn.subsystem
    id = rxn.id
    if subsys not in ["none", "", None] and id not in ["none", "", None]:
        for gene in rxn.genes:
            subsystem_dict[subsys].append(gene.id)
    for gene in rxn.genes:
        subsystem_dict["whole_metabolism"].append(gene.id)
logger.info("Finding KO divergence for all subsystems")
ko_divergence = metworkpy.divergence.ko_divergence(
    model=BASE_MODEL.copy(),
    genes_to_ko=GENE_LIST,
    target_networks=subsystem_dict,
    divergence_metric="kl",
    n_neighbors=N_NEIGHBORS,
    sample_count=NUM_SAMPLES,
    processes=PROCESSES,
)
logger.info("Found KO divergence, Saving to results dataframe")
gene_results_df = pd.concat([gene_results_df, ko_divergence], axis=1)

# Section: Knock out Max Flux
logger.info("Finding Max biomass flux following KO")
gene_deletion_df = cobra.flux_analysis.single_gene_deletion(BASE_MODEL)
# Convert from the frozen set to just the gene id
gene_deletion_df["ids"] = gene_deletion_df["ids"].apply(
    lambda x: x.__iter__().__next__()
)
gene_deletion_df = gene_deletion_df.set_index("ids")
gene_deletion_df = gene_deletion_df.rename({"growth": "ko_max_growth"}, axis=1)
gene_results_df = pd.concat(
    [gene_results_df, gene_deletion_df["ko_max_growth"]], axis=1
)

# Section: Reconcile Reaction based metrics with gene based metrics
# Start by creating a DataFrame, with gene and rxn columns
logger.info("Combining gene and reaction results dataframes")
genes_list = []
rxn_list = []
for rxn in BASE_MODEL.reactions:
    gene_set = rxn.genes
    for gene in gene_set:
        rxn_list.append(rxn.id)
        genes_list.append(gene.id)
gene_rxn_df = pd.DataFrame(
    {
        "gene": genes_list,
        "reaction": rxn_list,
    }
)
# Next, join the rxn_results_df to this
rxn_gene_results_df = (
    pd.merge(
        gene_rxn_df, rxn_results_df, how="left", left_on="reaction", right_index=True
    )
    .groupby("gene")
    .aggregate(
        {
            "pagerank": "max",
            "eigenvector": "max",
            "betweeness": "max",
            "closeness": "max",
            "reaction_pagerank": "max",
            "reaction_eigenvector_centrality": "max",
            "reaction_betweeness_centrality": "max",
            "reaction_closeness_centrality": "max",
            "fva_min": "min",
            "fva_max": "max",
        }
    )
)

results_df = pd.merge(
    gene_results_df, rxn_gene_results_df, how="left", left_index=True, right_index=True
)
logger.info("Saving results dataframe")
results_df.to_csv(RESULTS_PATH / "model_based_features.csv", index=True)
logger.info("Finished ;-)")
