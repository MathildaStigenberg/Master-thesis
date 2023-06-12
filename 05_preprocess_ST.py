#!/usr/bin/env python

## Preprocess ST data

# Import packages
import scanpy as sc
import numpy as np
import sys
import seaborn as sns
import matplotlib.pyplot as plt

# Path to concatenated ST file
concat_st = sys.argv[1]
# output folder
output = sys.argv[2]

# Read ST data
sdat = sc.read(concat_st)

# Make gene names unique
sdat.var_names_make_unique

# Check mitochondrial genes
sdat.var["mt"] = sdat.var_names.str.startswith("MT-")
# Calculate QC metric
sc.pp.calculate_qc_metrics(sdat, qc_vars=["mt"], inplace=True)

# Plot histograms of counts
#fig, axs = plt.subplots(1, 4, figsize=(15, 4))
#sns.distplot(sdat.obs["total_counts"], kde=False, ax=axs[0])
#sns.distplot(sdat.obs["total_counts"][sdat.obs["total_counts"] < 10000], kde=False, bins=40, ax=axs[1])
#sns.distplot(sdat.obs["n_genes_by_counts"], kde=False, bins=60, ax=axs[2])
#sns.distplot(sdat.obs["n_genes_by_counts"][sdat.obs["n_genes_by_counts"] < 4000], kde=False, bins=60, ax=axs[3])
#fig.savefig(output+'histogram_counts.png')

# Filter out spots and genes
sc.pp.filter_cells(sdat, min_counts=5000)
sc.pp.filter_cells(sdat, max_counts=35000)
sdat = sdat[sdat.obs["pct_counts_mt"] < 20]
sc.pp.filter_genes(sdat, min_cells=10)

# Keep only spots located in tissue
sdat = sdat[sdat.obs['in_tissue'] == 1]

# Normalize 
sdat = sdat.copy()
sdat.layers['normalized'] = sdat.X
sdat.X = sdat.layers['normalized']
sc.pp.log1p(sdat)

# G2M genes
g2m_genes_human=[ "HMGB2","CDK1", "NUSAP1" , "UBE2C","BIRC5","TPX2", "TOP2A", "NDC80","CKS2", "NUF2", "CKS1B","MKI67","TMPO", "CENPF"  ,"TACC3","FAM64A"  "SMC4", "CCNB2","CKAP2L"  "CKAP2","AURKB" , "BUB1", "KIF11","ANP32E" , "TUBB4B",  "GTSE1","KIF20B"  "HJURP",  "CDCA3","HN1",  "CDC20","TTK",  "CDC25C" , "KIF2C","RANGAP1","NCAPD2",  "DLGAP5",  "CDCA2","CDCA8","ECT2", "KIF23","HMMR","AURKA","PSRC1","ANLN", "LBR",  "CKAP5","CENPE","CTCF","NEK2", "G2E3", "GAS2L3" , "CBX5", "CENPA" ]

# Distribution of G2M per spot
g2m_genes_human = [i for i in g2m_genes_human if i in sdat.var_names]
g2m_per_spot = np.sum(sdat[:,np.array(g2m_genes_human)].X.copy(),axis=1)
sdat.obs['g2m_per_spot'] = g2m_per_spot

#fig, ax = plt.subplots()
#plt.hist(g2m_per_spot)
#plt.title('G2/M per spot')
#fig.savefig(output+'histogram_g2m.png')

#sc.pp.highly_variable_genes(sdat, subset=True, n_top_genes=1000)


# Write preprocessed anndata to file
sdat.write(output+'preprocessed_ST.h5ad')
