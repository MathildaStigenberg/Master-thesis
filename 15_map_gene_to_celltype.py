#Assign cell types to genes
import pickle 
import scanpy as sc
import numpy as np
WuID = 1
gene_celltype_mapping = dict()
for WuID in range(26):
    print(WuID)
    try:
        dat = sc.read_h5ad('/srv/mfs/hausserlab/data/sc_st_datasets_jakob/scRNAseq/dataReadyRAW_typed/Wu2021BreastCancer%04d.h5ad'%WuID)
    except:
        continue
    fraction_of_nan_celltypes = np.count_nonzero(np.array(dat.obs['cellTypeMajor']).astype('str')=='nan')/dat.shape[0]
    if fraction_of_nan_celltypes > 0.98:continue
    cat,count = np.unique(dat.obs['cellTypeMajor'].astype('str'),return_counts=True)
    if np.min(count)<=10:continue
    sc.pp.normalize_total(dat, target_sum=1e4)
    sc.pp.log1p(dat)
    #sc.pp.scale(dat, max_value=10)
    #sc.tl.pca(dat, svd_solver='arpack')
    #sc.pp.neighbors(dat, n_neighbors=10, n_pcs=40)
    #sc.tl.umap(dat)
    #sc.pl.umap(dat, color=['cellTypeMajor','celltype_census'])
    sc.tl.rank_genes_groups(dat, 'cellTypeMajor', method='t-test')
    #sc.pl.rank_genes_groups(dat, n_genes=10, sharey=False)
    celltypes = dat.obs['cellTypeMajor'].unique()
    celltypes = np.array(celltypes)
    celltypes = celltypes[1:] #Remove nan
    try:
        for gene in dat.var_names:
            scores = []
            for ct in celltypes:
                idx = np.where(dat.uns['rank_genes_groups']['names'][ct] == gene)[0][0]
                score = dat.uns['rank_genes_groups']['scores'][ct][idx]
                scores.append(score)
            if gene in gene_celltype_mapping.keys():
                gene_celltype_mapping[gene].append((celltypes[np.argmax(scores)],np.max(scores)))
            else:
                gene_celltype_mapping[gene] =[(celltypes[np.argmax(scores)],np.max(scores))]
    except:
        print('Something went wrong in sample', WuID)

    with open('celltype_mapping.pkl', 'wb') as f:
        pickle.dump(gene_celltype_mapping, f)
