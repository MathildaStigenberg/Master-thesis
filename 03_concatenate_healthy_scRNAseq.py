#!/usr/bin/env python

## Concatenate healthy samples

# Import packages 
import scanpy as sc
from anndata import AnnData
import anndata as ad
import sys

# Path to data folder
folder_data = sys.argv[1]
# Output 
output = sys.argv[2]

# Load scRNAseq data (anndata)
dat0 = sc.read_h5ad(folder_data+'GSM5022599_D1_filtered_feature_bc_matrix.h5ad')
dat1 = sc.read_h5ad(folder_data+'GSM5022600_D2_filtered_feature_bc_matrix.h5ad')
dat2 = sc.read_h5ad(folder_data+'GSM5022601_D3_filtered_feature_bc_matrix.h5ad')
dat3 = sc.read_h5ad(folder_data+'GSM5022602_D4_filtered_feature_bc_matrix.h5ad')
dat4 = sc.read_h5ad(folder_data+'GSM5022603_D5_filtered_feature_bc_matrix.h5ad')
dat5 = sc.read_h5ad(folder_data+'GSM5022606_D11_filtered_feature_bc_matrix.h5ad')

# Concatenate
healthy_all = ad.concat([dat0,dat1,dat2,dat3,dat4,dat5], join='inner', index_unique ='_')

# Write to file
healthy_all.write(output)
