#Classifier Codes

Requires:
- dataframes/
- graphs\_final/



mdr\_target.py 

* Creates target MDR JSON file for GNN training. 
* JSON file is already included in dataframes. This is for reference. 



graph\_classifier.py 

* Trains a GNN with graphs from graphs\_final/, target from mdr1000.json.
* Takes arguments 1, 2, 3, 4 for models GCN, GIN, GCN+Embed, GIN+Embed resepectively.
* Outputs PDF of training curve and final epoch metrics.
* Saves trained model to saved\_models/



node\_classifier.py 

* Trains a GNN with graphs from graphs\_final/, target from keio\_collection.xlsx
* Takes arguments 1, 2, 3, 4 for models GCN, GIN, GCN+Embed, GIN+Embed resepectively.
* Outputs PDF of training curve and final epoch metrics.
* Saves trained model to saved\_models/



g\_classifier\_umap.py

* Views UMAP projection of final graph representations from graph classifiers.
* Requires models to be trained and saved first.
* Takes arguments 1, 2, 3, 4 for models GCN, GIN, GCN+Embed, GIN+Embed resepectively.
* Forward pass inside must match model architecture.



n\_classifier\_umap.py

* Views UMAP projection of final node representations from node classifiers.
* Requires models to be trained and saved first.
* Takes arguments 1, 2, 3, 4 for models GCN, GIN, GCN+Embed, GIN+Embed resepectively.
* Forward pass inside must match model architecture.



classic\_ml.py

* Trains MLP and RF, both based on one-hot encoded a/p matrix
* Outputs a PDF for MLP loss and accuracy curve, MLP and RF final epoch metrics
* Saves y\_target and y\_output for both models, for use in ROC plotting.



roc.py 

* Outputs ROC figure comparing all models.
* Requires saved results from classic\_ml.py, and all 4 saved gnn models.





