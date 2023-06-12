#!/usr/bin/env python

## Concatenate scRNAseq samples

# Import packages
import scanpy as sc
from anndata import AnnData
import anndata as ad
import sys

# Path to data folder
folder_data = sys.argv[1]
# Output 
output = sys.argv[2]

# load scRNA sequencing data (anndata)
dat0 = sc.read_h5ad(folder_data + 'Wu2021BreastCancer0000.h5ad')
dat1 = sc.read_h5ad(folder_data + 'Wu2021BreastCancer0001.h5ad')
dat2 = sc.read_h5ad(folder_data + 'Wu2021BreastCancer0002.h5ad')
dat3 = sc.read_h5ad(folder_data + 'Wu2021BreastCancer0003.h5ad')
dat4 = sc.read_h5ad(folder_data + 'Wu2021BreastCancer0004.h5ad')
dat5 = sc.read_h5ad(folder_data + 'Wu2021BreastCancer0005.h5ad')
# dat6
dat7 = sc.read_h5ad(folder_data + 'Wu2021BreastCancer0007.h5ad')
# dat8
# dat9
dat10 = sc.read_h5ad(folder_data + 'Wu2021BreastCancer0010.h5ad')
dat11 = sc.read_h5ad(folder_data + 'Wu2021BreastCancer0011.h5ad')
# dat12
dat13 = sc.read_h5ad(folder_data + 'Wu2021BreastCancer0013.h5ad')
dat14 = sc.read_h5ad(folder_data + 'Wu2021BreastCancer0014.h5ad')
dat15 = sc.read_h5ad(folder_data + 'Wu2021BreastCancer0015.h5ad')
dat16 = sc.read_h5ad(folder_data + 'Wu2021BreastCancer0016.h5ad')
dat17 = sc.read_h5ad(folder_data + 'Wu2021BreastCancer0017.h5ad')
# dat18
dat19 = sc.read_h5ad(folder_data + 'Wu2021BreastCancer0019.h5ad')
# dat20
dat21 = sc.read_h5ad(folder_data + 'Wu2021BreastCancer0021.h5ad')
dat22 = sc.read_h5ad(folder_data + 'Wu2021BreastCancer0022.h5ad')
dat23 = sc.read_h5ad(folder_data + 'Wu2021BreastCancer0023.h5ad')
dat24 = sc.read_h5ad(folder_data + 'Wu2021BreastCancer0024.h5ad')
dat25 = sc.read_h5ad(folder_data + 'Wu2021BreastCancer0025.h5ad')

# Concatenate 
dat_all = ad.concat([dat0,dat1,dat2,dat3,dat4,dat5,dat7,dat10,dat11,dat13,dat14,dat15,dat16,dat17,dat19,dat21,dat22,dat23,dat24,dat25], join='inner', index_unique ='_')

# Write to file
dat_all.write(output)
