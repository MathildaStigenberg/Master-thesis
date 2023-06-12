# Import packages and load scRNA sequencing data (anndata)
import scanpy as sc
import textalloc as ta
from anndata import AnnData
import anndata as ad
import numpy as np
from numpy import nan
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from collections import OrderedDict
import sys
sys.path.insert(1,"/scratch/mathilda.stigenberg/TunedTumor/")

#Preprocessed cancer scRNAseq data
cancer_file = sys.argv[1]

#Preprocessed healthy scRNAseq data
healthy_file = sys.argv[2]

dat = sc.read_h5ad(cancer_file)
hdat = sc.read_h5ad(healthy_file)

output = sys.argv[3]

# G2M genes
g2m_genes_human=[ "HMGB2","CDK1", "NUSAP1" , "UBE2C","BIRC5","TPX2", "TOP2A", "NDC80","CKS2", "NUF2", "CKS1B","MKI67","TMPO", "CENPF"  ,"TACC3","FAM64A"  "SMC4", "CCNB2","CKAP2L"  "CKAP2","AURKB" , "BUB1", "KIF11","ANP32E" , "TUBB4B",  "GTSE1","KIF20B"  "HJURP",  "CDCA3","HN1", "CDC20","TTK", "CDC25C" , "KIF2C","RANGAP1","NCAPD2", "DLGAP5", "CDCA2","CDCA8","ECT2", "KIF23","HMMR","AURKA","PSRC1","ANLN", "LBR", "CKAP5","CENPE","CTCF","NEK2", "G2E3", "GAS2L3" , "CBX5", "CENPA" ]

# Find G2M related genes within cancer cells
cancer = dat[dat.obs.celltype_census == 'cancer cell                             ']

# Find G2M related genes within epithelial cells 
healthy = hdat[hdat.obs.celltype == 'luminal epithelial cell of mammary gland']

# Distribution of G2M per cancer cell
g2m_genes_cancer = [i for i in g2m_genes_human if i in cancer.var_names]
g2m_per_cell = np.sum(cancer[:,np.array(g2m_genes_cancer)].X.copy(),axis=1)
cancer.obs['g2m_per_cell'] = g2m_per_cell
fig, ax = plt.subplots()
fig.text(s='cancer subtype: ' + dat.uns['Subtype by IHC'], x=0.5, y=1, fontsize=12, ha='center')
plt.hist(g2m_per_cell)
plt.xlabel('G2M score')
plt.ylabel('count')
fig.savefig(output + 'g2m_per_cell_cancer.png', bbox_inches='tight')

# Distribution of G2M per healthy cell
g2m_genes_healthy = [i for i in g2m_genes_human if i in healthy.var_names]
g2m_per_cell = np.sum(healthy[:,np.array(g2m_genes_healthy)].X.copy(),axis=1)
healthy.obs['g2m_per_cell'] = g2m_per_cell
fig, ax = plt.subplots()
fig.text(s='healthy', x=0.5, y=1, fontsize=12, ha='center')
plt.hist(g2m_per_cell)
plt.xlabel('G2M score')
plt.ylabel('count')
fig.savefig(output + 'g2m_per_cell_healthy.png', bbox_inches='tight')

# Convert to dataframe
df = cancer.to_df(layer='normalized')
# Convert to dataframe
df_h = healthy.to_df(layer='normalized')

# Add G2M gene columns to a new dataframe
df_g2m = df[g2m_genes_cancer]
df_h_g2m = df_h[g2m_genes_healthy]

# Remove columns containing nan
df = df.dropna(axis=1)
df_h = df_h.dropna(axis=1)
df_g2m = df_g2m.dropna(axis=1)
df_h_g2m = df_h_g2m.dropna(axis=1)

# Remove all G2M genes
columns = list(df)
for i in g2m_genes_cancer:
    if i in columns:
        df.drop(i, axis=1, inplace=True)

columns_h = list(df_h)
for i in g2m_genes_healthy:
    if i in columns_h:
        df_h.drop(i, axis=1, inplace=True)

# Insert column with the G2M score
df.insert(loc = 0, column = 'g2m_per_cell', value = [i for i in cancer.obs['g2m_per_cell']])
df_h.insert(loc = 0, column = 'g2m_per_cell', value = [i for i in healthy.obs['g2m_per_cell']])
df_g2m.insert(loc = 0, column = 'g2m_per_cell', value = [i for i in cancer.obs['g2m_per_cell']])
df_h_g2m.insert(loc = 0, column = 'g2m_per_cell', value = [i for i in healthy.obs['g2m_per_cell']])

# Find correlated genes and remove nan values
columns = list(df)
correlation = {}
for col in columns:
    if col != 'g2m_per_cell':
        corr = np.corrcoef(df[col], df['g2m_per_cell'])
        correlation[col] = corr[0,1]

correlation = {k:v for k,v in correlation.items() if pd.Series(v).notna().all()}

# Find correlated genes and remove nan values 
columns_h = list(df_h)
correlation_h = {}
for col in columns_h:
    if col != 'g2m_per_cell':
        corr_h = np.corrcoef(df_h[col], df_h['g2m_per_cell'])
        correlation_h[col] = corr_h[0,1]

correlation_h = {k:v for k,v in correlation_h.items() if pd.Series(v).notna().all()}

# Calculate correlation for g2m genes and remove nan values 
columns_g2m = list(df_g2m)
correlation_g2m = {}
for col in columns_g2m:
    if col != 'g2m_per_cell':
        corr = np.corrcoef(df_g2m[col], df_g2m['g2m_per_cell'])
        correlation_g2m[col] = corr[0,1]

correlation_g2m = {k:v for k,v in correlation_g2m.items() if pd.Series(v).notna().all()}

# Calculate correlation for g2m genes and remove nan values 
columns_h_g2m = list(df_h_g2m)
correlation_h_g2m = {}
for col in columns_h_g2m:
    if col != 'g2m_per_cell':
        corr_h = np.corrcoef(df_h_g2m[col], df_h_g2m['g2m_per_cell'])
        correlation_h_g2m[col] = corr_h[0,1]

correlation_h_g2m = {k:v for k,v in correlation_h_g2m.items() if pd.Series(v).notna().all()}

# Distribution of correlated genes¨
fig, (ax1,ax2) = plt.subplots(1,2, figsize=(10,5))
sorted_correlation = OrderedDict(sorted(correlation.items(), key=lambda item: item[1], reverse=True))
bin_list = np.linspace(list(sorted_correlation.values())[-1],list(sorted_correlation.values())[0], 101)
fig.text(s='cancer subtype: ' + dat.uns['Subtype by IHC'], x=0.5, y=1, fontsize=12, ha='center')
ax1.hist(correlation.values(), alpha=0.5, bins=bin_list, label='cancer', density=True)
ax1.hist(correlation_h.values(), alpha=0.5, bins=bin_list, label='healthy', density=True)
ax1.set_xlabel('correlation')
ax1.set_ylabel('count')
ax1.legend(loc='upper right')
ax1.set_title('Distribution of correlated genes')

# Zoom in on the distribution 
ax2.hist(correlation.values(), alpha=0.5, bins=bin_list, label='cancer', density=True)
ax2.hist(correlation_h.values(), alpha=0.5, bins=bin_list, label='healthy', density=True)
ax2.set_xlabel('correlation')
ax2.set_ylabel('count')
ax2.legend(loc='upper right')
ax2.set_title('Zoom in on distribution of correlated genes')
plt.ylim(0,2)
fig.savefig(output + 'histogram_correlated_genes.png', bbox_inches='tight')

unique_cancer = list(correlation.keys()-correlation_h.keys())
unique_healthy = list(correlation_h.keys()-correlation.keys())
unique_g2m_cancer = list(correlation_g2m.keys()-correlation_h_g2m.keys())
unique_g2m_healthy = list(correlation_h_g2m.keys()-correlation_g2m.keys())

for i in unique_cancer:
    correlation_h[i] = 0
for i in unique_healthy:
    correlation[i] = 0
for i in unique_g2m_cancer:
    correlation_h_g2m[i] = 0
for i in unique_g2m_healthy:
    correlation_g2m[i] = 0

# Find x and y coordinates for the scatter plot (the correlation values of each gene for cancer and healthy patients)
common_genes = list(correlation.keys())
x = []
y = []
for i in common_genes:
    x.append(correlation_h[i])
    y.append(correlation[i])

common_g2m_genes = list(correlation_g2m.keys())
x_g2m = []
y_g2m = []
for i in common_g2m_genes:
    x_g2m.append(correlation_h_g2m[i])
    y_g2m.append(correlation_g2m[i])

# Scatter plot of correlation scores within cancer patients and healthy patients
fig, ax = plt.subplots()
plt.scatter(x, y, alpha=0.5, s=3, label='genes')
plt.scatter(x_g2m, y_g2m, alpha=0.5, s=3, c='red', label='G2M genes')
fig.text(s='cancer subtype: ' + dat.uns['Subtype by IHC'], x=0.5, y=1, fontsize=12, ha='center')
ax.spines.left.set_position('zero')
ax.spines.right.set_color('none')
ax.spines.bottom.set_position('zero')
ax.spines.top.set_color('none')
plt.grid()
plt.legend(loc='lower right', fontsize='x-small')
plt.xlabel('rG2M healthy',loc = 'right')
plt.ylabel('rG2M cancer', loc = 'top')
plt.title('Correlations of G2M related genes')
fig.savefig(output + 'healthy_vs_cancer_correlations.png', bbox_inches='tight')

# Create sorted dictionary and calculate difference in correlation between sick and healthy patients
sorted_genes = {}
for gene in correlation.keys():
    sorted_genes[gene] = correlation[gene] - correlation_h[gene]
sorted_genes = OrderedDict(sorted(sorted_genes.items(), key=lambda item: item[1], reverse=True))

# Find gene names of the 10 top sorted genes and 10 bottom sorted genes
genes = list(sorted_genes.keys())
labels = []
for i in genes[:10]:
    labels.append(i)
for i in genes[-10:]:
    labels.append(i)

# Find the x and y coordinates for those genes for the scatter plot
x_coord = []
y_coord = []
for i in labels:
    x_coord.append(correlation_h[i])
    y_coord.append(correlation[i])

# Scatter plot of correlation scores within cancer patients and healthy patients
fig, ax = plt.subplots()
ax.scatter(x, y, alpha=0.5, label='genes', s=3)
ax.scatter(x_g2m, y_g2m, alpha=0.5, c='red', label='G2M genes', s=3)
fig.text(s='cancer subtype: ' + dat.uns['Subtype by IHC'], x=0.5, y=1, fontsize=12, ha='center')
ta.allocate_text(fig,ax,x_coord,y_coord,
            labels,x_scatter=x,
            y_scatter=y,
            textsize=5.5,max_distance=0.2,
            min_distance=0.04,
            margin=0.039,
            linewidth=0.5,
            nbr_candidates=400)
ax.spines.left.set_position('zero')
ax.spines.right.set_color('none')
ax.spines.bottom.set_position('zero')
ax.spines.top.set_color('none')
plt.grid()
plt.legend(loc='lower right', fontsize='x-small')
plt.xlabel('rG2M healthy',loc = 'right')
plt.ylabel('rG2M cancer', loc = 'top')
plt.title('Correlations of G2M related genes')
fig.savefig(output + 'top_10_genes_cancer_vs_healthy.png', bbox_inches='tight')

# List of control genes
#genes = list(sorted_genes.keys())
#control = []
#for i in genes:
    #if i == 'CD274' or i == 'TGFB1' or i == 'TGFB2' or i == 'TGFB3' or i == 'TOP2A' or i =='TOP2B' or i == 'MKI67':
        #control.append(i)

# Find the x and y coordinates for those genes for the scatter plot
#x_cont = []
#y_cont = []
#for i in control:
    #x_cont.append(correlation_h[i])
    #y_cont.append(correlation[i])

#genes_g2m = list(correlation_g2m.keys())
#control_g2m = []
#for i in genes_g2m:
    #if i == 'CD274' or i == 'TGFB1' or i == 'TGFB2' or i == 'TGFB3' or i == 'TOP2A' or i =='TOP2B' or i == 'MKI67':
        #control_g2m.append(i)

# Find the x and y coordinates for those genes for the scatter plot
#x_cont_g2m = []
#y_cont_g2m = []
#for i in control_g2m:
   # x_cont_g2m.append(correlation_h_g2m[i])
    #y_cont_g2m.append(correlation_g2m[i])

# Scatter plot of correlation scores within cancer patients and healthy patients
#fig, ax = plt.subplots()
#ax.scatter(x, y, alpha=0.5, label='genes', s=3)
#ax.scatter(x_g2m, y_g2m, alpha=0.5, c='red', label='G2M genes', s=3)
#ax.scatter(x_cont,y_cont, alpha=0.5, c='orange', label='control genes', s=4)
#ax.scatter(x_cont_g2m,y_cont_g2m, alpha=0.5, c='orange', s=4)
#fig.text(s='cancer subtype: ' + dat.uns['Subtype by IHC'], x=0.5, y=1, fontsize=12, ha='center')
#ta.allocate_text(fig,ax,x_cont_g2m,y_cont_g2m,
            #control_g2m,x_scatter=x,
            #y_scatter=y,
            #textsize=5.5,max_distance=0.2,
            #min_distance=0.04,
            #margin=0.039,
            #linewidth=0.5,
            #nbr_candidates=400)
#ax.spines.left.set_position('zero')
#ax.spines.right.set_color('none')
#ax.spines.bottom.set_position('zero')
#ax.spines.top.set_color('none')
#ta.allocate_text(fig,ax,x_cont,y_cont,
            #control,x_scatter=x,
            #y_scatter=y,
            #textsize=5.5,max_distance=0.2,
            #min_distance=0.04,
            #margin=0.039,
            #linewidth=0.5,
            #nbr_candidates=400)
#plt.grid()
#plt.legend(loc='lower right', fontsize='x-small')
#plt.xlabel('rG2M healthy',loc = 'right')
#plt.ylabel('rG2M cancer', loc = 'top')
#plt.title('Correlations of G2M related genes')
#fig.savefig(output + 'control_genes.png', bbox_inches='tight')

# A list of genes which are correlated and are to be excluded from the spatial transcriptomics data
th_y = 0.1
th_x = 0
genes_to_exclude = []
for i in correlation:
    if correlation[i] < 0.1+th_y and correlation_h[i] > 0.1+th_x or correlation[i] > -0.1-th_y and correlation_h[i] < -0.1-th_x:
        genes_to_exclude.append(i)
    elif correlation[i] > 0.1+th_y and correlation_h[i] > 0.1+th_x or correlation[i] < -0.1-th_y and correlation_h[i] < -0.1-th_x:
        if correlation[i] >= 0 and correlation_h[i] >= 0:
            if correlation[i]+th_x < correlation_h[i]+th_y or correlation[i]-th_x == correlation_h[i]+th_y:
                genes_to_exclude.append(i)
        if correlation[i] < 0 and correlation[i] < 0:
            if correlation[i]-th_x > correlation_h[i]-th_y or correlation[i]-th_x == correlation_h[i]-th_y:
                genes_to_exclude.append(i)

# x and y coordinates of the excluded genes for the scatter plot
y_e = []
x_e = []
for i in genes_to_exclude:
    y_e.append(correlation[i])
    x_e.append(correlation_h[i])

# Scatter plot of correlation scores within cancer patients and healthy patients as well as the excluded genes
fig, ax = plt.subplots()
ax.scatter(x, y, alpha=0.5, label='genes', s=3)
ax.scatter(x_g2m, y_g2m, alpha=0.5, c='red', label='G2M genes', s=3)
ax.scatter(x_e,y_e, alpha=0.5, c='black', label='excluded genes', s=3)
fig.text(s='cancer subtype: ' + dat.uns['Subtype by IHC'], x=0.5, y=1, fontsize=12, ha='center')
ax.spines.left.set_position('zero')
ax.spines.right.set_color('none')
ax.spines.bottom.set_position('zero')
ax.spines.top.set_color('none')
plt.grid()
plt.legend(loc='lower right', fontsize='x-small')
plt.xlabel('rG2M healthy',loc = 'right')
plt.ylabel('rG2M cancer', loc = 'top')
plt.title('Correlations of G2M related genes')
fig.savefig(output + 'genes_to_exclude.png', bbox_inches='tight')

# A list of genes to exclude in the spatial data
genes_to_exclude = genes_to_exclude + g2m_genes_human

# File containing genes to exclude
with open(output + 'exclude_genes.txt', 'w') as f:
    for i in genes_to_exclude:
        f.write(i)
        f.write('\n')

# Save sorted list in a .rnk-file for GSEA analysis
with open(output + 'sorted-list.rnk', 'w') as f:
    for k,v in sorted_genes.items():
        f.write(k)
        f.write('\t')
        f.write(str(v))
        f.write('\n')
