import pickle
import sys

# the cell type mapping dictionary obtained from the script map_gene_to_celltype
cell_types_samples = sys.argv[1]

# top correlated genes for every latent variable
top_corr_genes = sys.argv[2]

# bottom correlated genes for every latent variable
bottom_corr_genes = sys.argv[3]

output = sys.argv[4]

with open(cell_types_samples, 'rb') as f:
    celltypes = pickle.load(f)

for i in celltypes.keys():
    celltypes[i].sort(key=lambda tup: tup[1], reverse=True)

with open(top_corr_genes, 'rb') as f:
    top = pickle.load(f)

with open(bottom_corr_genes, 'rb') as f:
    bot = pickle.load(f)

top_celltypes = {}
bottom_celltypes = {}

for i in range(0,20):
    top_celltypes[i] = []
    bottom_celltypes[i] = []

for l in range(0,20):
    for i in map(lambda x: x[0], top[l]):
        if i in celltypes.keys():
            top_celltypes[l].append((i, celltypes[i][0][0]))
        else:
            top_celltypes[l].append((i, 'unknown'))
    for i in map(lambda x: x[0], bot[l]):
        if i in celltypes.keys():
            bottom_celltypes[l].append((i, celltypes[i][0][0]))
        else:
            bottom_celltypes[l].append((i, 'unknown'))

with open(output+'bottom_cell_types.pkl', 'wb') as l:
    pickle.dump(bottom_celltypes, l)

with open(output+'top_cell_types.pkl', 'wb') as l:
    pickle.dump(top_celltypes, l)
