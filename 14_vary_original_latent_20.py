from tensorflow.keras.models import load_model
from tensorflow import keras
from keras import backend as K
from collections import OrderedDict
import sklearn
from sklearn.preprocessing import MinMaxScaler, normalize
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import scanpy as sc
import pandas as pd
from statistics import mean
from scipy.stats import pearsonr
from collections import deque
import subprocess
from gprofiler import gprofiler
import pickle

# output folder
output = sys.argv[1]

# trained model with 20 latent variables
encoder = load_model(sys.argv[2])
decoder = load_model(sys.argv[3])
supervised = load_model(sys.argv[4])

# spatial data
dat = sc.read_h5ad(sys.argv[5])

df = dat.to_df(layer='normalized')

ex_genes = sys.argv[6]

# Exclude G2M related genes
genes_to_exclude = open(ex_genes)
for gene in genes_to_exclude:
    for i in gene.split('\n'):
        if i in list(df.columns):
            df.drop(i, axis=1, inplace=True)

df.insert(loc = 0, column = 'g2m_per_spot', value = [i for i in dat.obs['g2m_per_spot']])

genes = list(df.columns)[1:]

gene_index = {}
v = 0
for i in genes:
    gene_index[i] = v
    v += 1

X = df.iloc[:,1:]
Y = df.iloc[:,0]

X = normalize(X, norm='l2')

scaler = MinMaxScaler()
scaler.fit(X)
X = scaler.transform(X)

# shuffle X and Y
X = sklearn.utils.shuffle(X, random_state=42)
Y = sklearn.utils.shuffle(Y, random_state=42)

latent = encoder.predict(X)[2]

recon = decoder.predict(latent)

growth = supervised.predict(latent)

growth = growth.T.flatten()

sorted_top_genes_lv = {}
sorted_bottom_genes_lv = {}

for i in range(0,latent.shape[1]):
    sorted_top_genes_lv[i] = deque()
    sorted_bottom_genes_lv[i] = deque()

matrix = {}
matrix['Latent variable'] = []
matrix['slopeGrowth'] = []
matrix['Upregulated genes'] = []
matrix['Downregulated genes'] = []

genes_stats = {}
for i in range(0,latent.shape[1]):
    genes_stats[i] = []

growth_stats = []

for l in range(0,latent.shape[1]):
    genes_corr = {}
    latent_values = latent[:,l]
    corr, pvalue = pearsonr(latent_values, growth)
    slope_growth = np.polyfit(latent_values, growth, 1)[0]
    growth_stats.append([corr, pvalue, slope_growth])
    for i, g in enumerate(genes):
        gene_exp = recon[:,i]
        corr, pvalue = pearsonr(latent_values,gene_exp)
        slope_gene = np.polyfit(latent_values, gene_exp,1)[0]
        range_exp = np.max(gene_exp)- np.min(gene_exp)
        genes_stats[l].append((g,slope_gene,corr,pvalue, range_exp))
        genes_corr[g] = corr
    genes_stats[l].sort(key=lambda tup: tup[2], reverse=True)
    sorted_corr_genes = OrderedDict(sorted(genes_corr.items(), key=lambda item: item[1], reverse=True))
    low, high = np.percentile(list(sorted_corr_genes.values()),[10,90])
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    for k, v in sorted_corr_genes.items():
        if v <= low:
            gene_exp = recon[:,gene_index[k]]
            slope_gene = np.polyfit(latent_values, gene_exp,1)[0]
            corr, pvalue = pearsonr(latent_values,gene_exp)
            range_exp = np.max(gene_exp)- np.min(gene_exp)
            sorted_bottom_genes_lv[l].appendleft((k, v, slope_gene, pvalue, range_exp))
        if v >= high:
            gene_exp = recon[:,gene_index[k]]
            slope_gene = np.polyfit(latent_values, gene_exp,1)[0]
            corr, pvalue = pearsonr(latent_values,gene_exp)
            range_exp = np.max(gene_exp)- np.min(gene_exp)
            sorted_top_genes_lv[l].append((k, v, slope_gene, pvalue, range_exp))
    ax1.scatter(latent_values, recon[:,gene_index[k]], alpha=0.5)
    ax2.scatter(growth, recon[:,gene_index[k]], alpha=0.5)
    ax1.set_title('Gene expression when varying a latent variable')
    ax1.set_xlabel('Latent variable {}'.format(l))
    ax1.set_ylabel('Gene expression')
    fig1.savefig(output+'gene_vs_vary_latent_{}.png'.format(l))
    ax2.set_title('Growth vs. gene expression when varying latent variable {}'.format(l))
    ax2.set_xlabel('Growth')
    ax2.set_ylabel('Gene expression')
    fig2.savefig(output+'gene_vs_growth_{}.png'.format(l))
    fig3, ax3 = plt.subplots()
    ax3.scatter(latent_values, growth, alpha=0.5)
    ax3.set_title('Growth when varying a latent variable')
    ax3.set_xlabel('Latent variable {}'.format(l))
    ax3.set_ylabel('Growth')
    fig3.savefig(output+'growth_vs_vary_latent_{}.png'.format(l))
    matrix['Latent variable'] += [l]
    matrix['slopeGrowth'] += [round(slope_growth,4)]

with open(output+'growth_stats.pkl', 'wb') as l:
    pickle.dump(growth_stats, l)

with open(output+'genes_stats.pkl', 'wb') as l:
    pickle.dump(genes_stats, l)

top_genes_lv = {}
bottom_genes_lv = {}

for i in range(0,latent.shape[1]):
    top_genes_lv[i] = set(map(lambda x: x[0], sorted_top_genes_lv[i]))
    bottom_genes_lv[i] = set(map(lambda x: x[0], sorted_bottom_genes_lv[i]))

unique_up = {}
unique_down = {}

for i in range(0, latent.shape[1]):
    unique_up[i] = top_genes_lv[i].difference(top_genes_lv[(i+1)%20], top_genes_lv[(i+2)%20], top_genes_lv[(i+3)%20], top_genes_lv[(i+4)%20], top_genes_lv[(i+5)%20], top_genes_lv[(i+6)%20], top_genes_lv[(i+7)%20], top_genes_lv[(i+8)%20], top_genes_lv[(i+9)%20], top_genes_lv[(i+10)%20], top_genes_lv[(i+11)%20], top_genes_lv[(i+12)%20], top_genes_lv[(i+13)%20], top_genes_lv[(i+14)%20], top_genes_lv[(i+15)%20], top_genes_lv[(i+16)%20], top_genes_lv[(i+17)%20], top_genes_lv[(i+18)%20], top_genes_lv[(i+19)%20])
    unique_down[i] = bottom_genes_lv[i].difference(bottom_genes_lv[(i+1)%20], bottom_genes_lv[(i+2)%20], bottom_genes_lv[(i+3)%20], bottom_genes_lv[(i+4)%20], bottom_genes_lv[(i+5)%20], bottom_genes_lv[(i+6)%20], bottom_genes_lv[(i+7)%20], bottom_genes_lv[(i+8)%20], bottom_genes_lv[(i+9)%20], bottom_genes_lv[(i+10)%20], bottom_genes_lv[(i+11)%20], bottom_genes_lv[(i+12)%20], bottom_genes_lv[(i+13)%20], bottom_genes_lv[(i+14)%20], bottom_genes_lv[(i+15)%20], bottom_genes_lv[(i+16)%20], bottom_genes_lv[(i+17)%20], bottom_genes_lv[(i+18)%20], bottom_genes_lv[(i+19)%20])

sorted_unique_up = {}
sorted_unique_down = {}

for i in range(0,latent.shape[1]):
    sorted_unique_up[i] = []
    sorted_unique_down[i] = []

for i in range(0,latent.shape[1]):
    for g in map(lambda x: x[0], sorted_top_genes_lv[i]):
        if g in unique_up[i]:
            sorted_unique_up[i].append(g)
    for g in map(lambda x: x[0], sorted_bottom_genes_lv[i]):
        if g in unique_down[i]:
            sorted_unique_down[i].append(g)

top_stats = {}
bottom_stats = {}

for i in range(0, latent.shape[1]):
    top_stats[i] = []
    bottom_stats[i] = []

for i in range(0,latent.shape[1]):
    for s in sorted_bottom_genes_lv[i]:
        if s[0] in sorted_unique_down[i]:
            bottom_stats[i].append([s[0], s[1], s[2], s[3], s[4]])
    for s in sorted_top_genes_lv[i]:
        if s[0] in sorted_unique_up[i]:
            top_stats[i].append([s[0], s[1], s[2], s[3], s[4]])

with open(output+'bottom_genes_stats.pkl', 'wb') as l:
    pickle.dump(bottom_stats, l)

with open(output+'top_genes_stats.pkl', 'wb') as l:
    pickle.dump(top_stats, l)

for i in range(0,latent.shape[1]):
    matrix['Upregulated genes'].append([sorted_unique_up[i]])
    matrix['Downregulated genes'].append([sorted_unique_down[i]])

upregulated_id = []
downregulated_id = []
upregulated_name = []
downregulated_name = []
for i in range(0,latent.shape[1]):
    kegg_up = []
    kegg_down = []
    if len(sorted_unique_up[i]) == 0:
        upregulated_id.append([('None')])
        upregulated_name.append([('None')])
    else:
        up_enrichment = gprofiler(query=sorted_unique_up[i], organism='hsapiens', ordered_query=True)
        if up_enrichment is not None:
            upregulated_id.append(list(up_enrichment.sort_values('p.value')['term.id']))
            kegg = up_enrichment.sort_values('p.value').loc[up_enrichment['term.id'].str.get(0) == 'K', 'term.name']
            if len(kegg) == 0:
                upregulated_name.append([('None')])
            else:
                for p in range(0,len(kegg)):
                    kegg_up.append(kegg.iloc[p])
                upregulated_name.append(kegg_up)
        else:
            upregulated_id.append([('None')])
            upregulated_name.append([('None')])
    if len(sorted_unique_down[i]) == 0:
        downregulated_id.append([('None')])
        downregulated_name.append([('None')])
    else:
        down_enrichment = gprofiler(query=sorted_unique_down[i], organism='hsapiens', ordered_query=True)
        if down_enrichment is not None:
            downregulated_id.append(list(down_enrichment.sort_values('p.value')['term.id']))
            kegg = down_enrichment.sort_values('p.value').loc[down_enrichment['term.id'].str.get(0) == 'K', 'term.name']
            if len(kegg) == 0:
                downregulated_name.append([('None')])
            else:
                for p in range(0,len(kegg)):
                    kegg_down.append(kegg.iloc[p])
                downregulated_name.append(kegg_down)
        else:
            downregulated_id.append([('None')])
            downregulated_name.append([('None')])

up_go_terms = {}
down_go_terms = {}
for i in range(0,latent.shape[1]):
    up_go_terms[i] = []
    down_go_terms[i] = []
for i in range(0,latent.shape[1]):
    up_string_go_terms = ''.join(map(str,'\n'.join(filter(lambda x: x[0] == 'G', upregulated_id[i])))).split('\n')
    up_go_terms[i] += up_string_go_terms
    down_string_go_terms = ''.join(map(str,'\n'.join(filter(lambda x: x[0] == 'G', downregulated_id[i])))).split('\n')
    down_go_terms[i] += down_string_go_terms

with open(output+'upregulated_GO_terms.txt', 'w') as f:
    for k,v in up_go_terms.items():
        f.write('upregulated GO terms for latent variable {}'.format(k))
        f.write('\n')
        for i in v:
            f.write(i)
            f.write('\n')

with open(output+'downregulated_GO_terms.txt', 'w') as f:
    for k,v in down_go_terms.items():
        f.write('downregulated GO terms for latent variable {}'.format(k))
        f.write('\n')
        for i in v:
            f.write(i)
            f.write('\n')

for key, value in up_go_terms.items():
   subprocess.run(['/scratch/jakob.rosenbauer/miniconda/envs/dwls/bin/Rscript','/scratch/jakob.rosenbauer/TunedTumor_ST/RunRevigo.R',output+'up_{}.pdf'.format(key),*value])
for key, value in down_go_terms.items():
   subprocess.run(['/scratch/jakob.rosenbauer/miniconda/envs/dwls/bin/Rscript','/scratch/jakob.rosenbauer/TunedTumor_ST/RunRevigo.R',output+'down_{}.pdf'.format(key),*value])

table = pd.DataFrame(matrix)

pd.set_option("display.max_colwidth", None)

table['Upregulated term.id'] = upregulated_id
table['Downregulated term.id'] = downregulated_id
table['Upregulated KEGG'] = upregulated_name
table['Downregulated KEGG'] = downregulated_name

for i in table.columns:
    table[i] = table[i].astype(str).replace(to_replace='[[]', value='', regex=True)
    table[i] = table[i].astype(str).replace(to_replace='[]]', value='', regex=True)
    table[i] = table[i].str.replace("'", "")

table.to_pickle(output+'pathways_table_20.pkl')
