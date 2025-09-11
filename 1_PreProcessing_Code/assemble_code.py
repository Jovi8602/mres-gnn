import subprocess
import glob
import pandas as pd
import os
from Bio import SeqIO

def run_ragtag(reference_fasta, query_fasta, output_dir, threads=4):
    cmd = [
        "ragtag.py", "scaffold",
        reference_fasta,
        query_fasta,
        "-o", output_dir,
        "-t", str(threads)
    ]
    subprocess.run(cmd, check=True)
    print(f"RagTag completed: output in {output_dir}")


# Constructing full assemblies from contigs
def full_assembly(contig_file, output_name):
    with open(f'temp_forshow/{output_name}', "w") as output:
        full_sequence = ""
        for record in SeqIO.parse(contig_file, "fasta"):
            full_sequence += str(record.seq)

        output.write(f">{output_name}\n{full_sequence}\n")

    print(f"Assembled genome saved: {output_name}")

def assemble():
    file_list = glob.glob('collection1000/*')
    enterodata = pd.read_csv('dataframes/entero_metadata.csv')
    metadata = (pd.read_csv('dataframes/metadata.csv')).dropna(subset=["name_in_presence_absence"])
    reference = 'dataframes/ecoli_reference.fasta'

    for file in file_list:
        contig_file = f"collection1000/{os.path.basename(file)}/{os.path.basename(file)}"

        #Converting Enterobase assembly ID to Strain ID format in Metadata.
        contig = os.path.basename(file).split(".")[0]
        ass_id = contig.split("_genomic")[0]
        barcode = enterodata.loc[enterodata['Assembly barcode'] == ass_id, 'Barcode'].item()
        strain = metadata.loc[metadata['ID'].str.startswith(barcode), 'ID'].item()
        assembly = f"{strain.upper()}.fasta"

        run_ragtag(reference, contig_file, output_dir=f"ragtag/{assembly}")

    #Assembling from ragtag output
    file_list = glob.glob('ragtag/*')

    for file in file_list:
        preassembly = f'{file}/ragtag.scaffold.fasta'
        assembly = os.path.basename(file)
        full_assembly(preassembly, assembly)

    print(len(file_list), 'files downloaded and assembled.')

if __name__ == "__main__":
    assemble()
