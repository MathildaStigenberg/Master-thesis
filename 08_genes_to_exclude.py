# Import packages
import numpy as np
import sys
sys.path.insert(1,"/scratch/mathilda.stigenberg/TunedTumor/")

#Output file containing genes to exclude
output = sys.argv[1]

# Exclude G2M related genes
genes_excluded = set()
for n in range(0,26):
    if n != 6 and n != 9 and n != 12 and n != 16:
        with open(f'../results/31_genes_to_exclude/{n}_exclude.txt', 'r') as file:
            for gene in file:
                gene = gene.strip()
                for i in gene.split('\n'):
                    genes_excluded.add(i)

with open(output, 'w') as f:
    for i in genes_excluded:
        f.write(i)
        f.write('\n')
