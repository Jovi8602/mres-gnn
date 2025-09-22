# Data Availability
Several folders are too large to upload to GitHub, but can be downloaded from the google drive link below: 

[link
](https://drive.google.com/drive/folders/1bUzDr6qvss4kbZKA-cRYjEh5292Q4ed9?usp=drive_link)

Download and extract the folders so the directory structure looks spmething like this:

project_root/
- dataframes/
- graphs_final/
- 3_Classifier_Code/
- 4_VGAE_Code/
- ...

# Script Usage
All scripts have to be moved from their folders out to the root folder, to utilise dataframes/, results/, graphs\_final/, or saved\_models/.

Directory structure should look something like this: 

project_root/
- dataframes/
- graphs_final/
- saved\_models/
- n_classifier_umap.py
- vgae_sampling.py
- ...


General Instructions are provided below. 
Each section can be run individually, prodivded all external data folders on Google Drive are downloaded. 

Preprocessing: 
1) assemble_code.py using collection1000/ to create assembled/, containing finished assemblies for all 1000 strains. 
2) gene_lists.py to create filtered_genes/, containing respective gene FASTA files for each strain. 
3) blast_code.py using assembled/ and filtered_genes/ to create results/, containing gene information spreadsheets for each strain. 

Graph Building:
1) graph_code.py to create graphs_final/, containing all 1000 graph .pkl files. 
2) graph_drawing.py using one of any folder in results/ to visualise a graph, saved as PNG.
3) graph_stats.py to create MDR_graph_stats.pdf, for graph statistic comparison between MDR+ and MDR- genomes. 

Classifiers:
- Graph
1) graph_classifier.py to to train a specified MDR classification model, saved to saved_models/. Training results and metrics are saved as PDF. 
2) g_classifier_umap.py for graph UMAP projections PNG using specified model, saved as PNG. 
3) classic_ml.py to train RF and MLP models. Training results are saved as PDF. 
4) roc.py to produce ROC curves plot, saved as PNG. Requires all 4 MDR GNN models to be first trained. 
- Node
1) node_classifier.py to to train a specified essentiality classification model, saved to saved_models/. Training results and metrics are saved as PDF. 
2) n_classifier_umap.py for node UMAP projections using specified model, saved as PNG. 

VGAE:
1) vgae.py to train VGAE model, saved to saved_models/. Training results saved to PDF. 
2) vgae_umap.py for node embedding UMAP projections.
3) vgae_sampling.py to create a synthetic graph, saved as PNG. 


Each subfolder has READMEs of their own to explain what each script does. 






