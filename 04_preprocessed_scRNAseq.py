#!/usr/bin/env python

## Preprocess scRNAseq samples

# Import packages
import scanpy as sc
import numpy as np
import sys
sys.path.insert(1,"/scratch/mathilda.stigenberg/TunedTumor/")
from loadData import loadWuSc

# scRNAseq file
sc_file = sys.argv[1]

# output file
output = sys.argv[2]

sample, dat = sc.read_h5ad(sc_file)

# Use the raw counts 
dat.X = dat.layers['raw']

# Uniquely name the variable names (genes)
dat.var_names_make_unique()

# Mitochondrial genes
dat.var["mt"] = dat.var_names.str.startswith("MT-")
# Ribosomal genes
dat.var["ribo"] = dat.var_names.str.startswith(("RPS", "RPL"))
# Hemoglobin genes
dat.var["hb"] = dat.var_names.str.startswith(("^HB[^(P)]"))

# Calculate QC metrics 
sc.pp.calculate_qc_metrics(
    dat, qc_vars=["mt", "ribo", "hb"], inplace=True, percent_top=[20], log1p=True
)

# Function on how to define an outlier 
def is_outlier(dat, metric: str, nmads: int):
    M = dat.obs[metric]
    outlier = (M < np.median(M) - nmads * M.mad()) | (                
        np.median(M) + nmads * M.mad() < M
    )
    return outlier

# Add column to obs which displays if cells are outliers or not based on the covariates. 
dat.obs["outlier"] = (
    is_outlier(dat, "log1p_total_counts", 5)
    | is_outlier(dat, "log1p_n_genes_by_counts", 5)
    | is_outlier(dat, "pct_counts_in_top_20_genes", 5)
)
dat.obs.outlier.value_counts()

# Filter cells based on the mitochondrial content
dat.obs["mt_outlier"] = is_outlier(dat, "pct_counts_mt", 3) | (
    dat.obs["pct_counts_mt"] > 8
)
dat.obs.mt_outlier.value_counts()

# Remove the outliers 
dat = dat[(~dat.obs.outlier) & (~dat.obs.mt_outlier)].copy()

# Filter out genes not detected in at least 20 cells
sc.pp.filter_genes(dat, min_cells=20)

# Filter cells with at most 200 genes
dat = dat.copy()
sc.pp.filter_cells(dat, min_genes=200)

# Normalize
dat = dat.copy()
dat.layers['normalized'] = dat.X
dat.X = dat.layers['normalized']
sc.pp.normalize_total(dat, target_sum=1e6)

# Write to file
dat.write(output)
