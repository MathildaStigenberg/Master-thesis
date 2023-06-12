#!/usr/bin/env python

## Concatenate ST samples

# Import packages
import scanpy as sc
from anndata import AnnData
import anndata as ad
import sys
sys.path.insert(1,"/scratch/mathilda.stigenberg/TunedTumor/")
from loadData import read_visium

# Path to data folder
folder_data = sys.argv[1]
# Output 
output = sys.argv[2]


# Load ST data (anndata)
sdat0 = read_visium(path_counts=folder_data+'raw_count_matrices/'+'1142243F'+'_raw_feature_bc_matrix', load_images=True,path_spatial=folder_data+'spatial/'+'1142243F'+'_spatial',metadata_path=folder_data+'metadata/'+'1142243F'+'_metadata.csv')
sdat1 = read_visium(path_counts=folder_data+'raw_count_matrices/'+'1160920F'+'_raw_feature_bc_matrix', load_images=True,path_spatial=folder_data+'spatial/'+'1160920F'+'_spatial',metadata_path=folder_data+'metadata/'+'1160920F'+'_metadata.csv')
sdat2 = read_visium(path_counts=folder_data+'raw_count_matrices/'+'CID4290'+'_raw_feature_bc_matrix', load_images=True,path_spatial=folder_data+'spatial/'+'CID4290'+'_spatial',metadata_path=folder_data+'metadata/'+'CID4290'+'_metadata.csv')
sdat3 = read_visium(path_counts=folder_data+'raw_count_matrices/'+'CID4465'+'_raw_feature_bc_matrix', load_images=True,path_spatial=folder_data+'spatial/'+'CID4465'+'_spatial',metadata_path=folder_data+'metadata/'+'CID4465'+'_metadata.csv')
sdat4 = read_visium(path_counts=folder_data+'raw_count_matrices/'+'CID44971'+'_raw_feature_bc_matrix', load_images=True,path_spatial=folder_data+'spatial/'+'CID44971'+'_spatial',metadata_path=folder_data+'metadata/'+'CID44971'+'_metadata.csv')
sdat5 = read_visium(path_counts=folder_data+'raw_count_matrices/'+'CID4535'+'_raw_feature_bc_matrix', load_images=True,path_spatial=folder_data+'spatial/'+'CID4535'+'_spatial',metadata_path=folder_data+'metadata/'+'CID4535'+'_metadata.csv')

# Concatenate 
sdat_all = ad.concat([sdat0,sdat1,sdat2,sdat3,sdat4,sdat5], join='inner', index_unique ='_')

# Write to file
sdat_all.write(output)
