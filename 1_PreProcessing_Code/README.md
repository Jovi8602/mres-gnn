Requires:

dataframes/

collection1000/ - Raw assemblies from Enterobase

assembled/ - Created by assemble\_code

filtered\_genes/ - Created by gene\_lists





selecting\_code.py - Code for selecting strains and preparing esc\_strains.txt. Does not reproduce 1000 chosen strains.

assemble\_code.py - Takes all raw assemblies from collection1000/, saves all 1000 final assemblies into assembled/.

gene\_lists.py - Creates list of genes sequences for each 1000 strain. Saves to filtered\_genes/.

blast\_code.py - BLAST search using filtered\_genes/ and assembled/. BLAST results for each strain is saved in results/.

esc\_strains.txt - List of strains to be used for search in Enterobase.

