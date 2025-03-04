# Script for processing protein sequences and predicting
# different chemical properties

# Setup
# Import Needed Libraries
library(here)
library(tidyverse)
library(Peptides)

# Read the file with all the sequences
protein_info <- read_csv(here("kinase_target_ml", "data", "uniprot_cleaned.csv")) |>
  mutate(net_charge = charge(seq = Sequence)) |>
  mutate("boman" = boman(Sequence)) |> 
  mutate("hmoment" = hmoment(Sequence)) |> 
  mutate("hydrophobicity" = hydrophobicity(Sequence))  |> 
  mutate("instaIndex" = instaIndex(Sequence)) |>
  mutate("isoelectric_point" = pI(Sequence)) 


# Save the results to a new file
protein_info |> write_csv(here("kinase_target_ml", "results", "protein_info.csv"))
