Requires:

dataframes/

graphs\_final/



vgae.py -

* Trains a VGAE model using graphs\_final/
* Only GIN encoder tried
* Only dot decoder and bilinear decoder within
* Saves trained model states to saved\_models/



vgae\_umap.py -

* Plots UMAP latent space using saved model state
* Uses latent vectors across a set number of graphs
* Color codes and saves png for essentiality, gene length, gene identity



vgae\_sampling.py -

* Code for sampling random vectors and passing them through a decoder
* Number of nodes can be controlled
* Contains a logic for labelling nodes based on nearest gene in latent space (nearest centroid)
* Saves a PNG of the synthetic graph
