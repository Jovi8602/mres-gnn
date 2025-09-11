import glob
import os
import pandas as pd
from Bio import SeqIO

def main():
    metadata = (pd.read_csv('dataframes/metadata.csv')).dropna(subset=["name_in_presence_absence"])
    presence_data = pd.read_csv('dataframes/presence.csv')

    # Making custom gene fasta file for every single strain.
    files = glob.glob('assembled/*')
    strains = []
    matrix_names = []
    for file in files:
        str = os.path.basename(file).split('.')[0]
        strains.append(str)
        matrix = metadata.loc[metadata['ID'] == str, 'name_in_presence_absence'].item()
        matrix_names.append(matrix)

    presence_dict_1000 = {}
    all_genes = list(presence_data['Strain'])
    del all_genes[0]  #Remove lineage header

    for i, genome in enumerate(matrix_names):
        genes_here = []
        abs_list = list(presence_data[genome])
        del abs_list[0]  #Remove lineage value
        for u in range(len(abs_list)):
            if abs_list[u] == 1:
                genes_here.append(all_genes[u])

        full_file = "dataframes/all_genes.fa"
        output_fasta = f'filtered_genes/{strains[i]}_genes.fasta'

        seq_dict = {}

        for record in SeqIO.parse(full_file, "fasta"):
            gene_id = record.id.split()[0]
            seq_dict[gene_id] = record

        with open(output_fasta, "w") as output:
            for gene in genes_here:
                if gene in seq_dict:
                    SeqIO.write(seq_dict[gene], output, "fasta")

        print(len(genes_here))
        print(f'Matrix Name: {genome} Actual Name: {strains[i]}. Genes fasta file created.')

if __name__ == "__main__":
    main()
