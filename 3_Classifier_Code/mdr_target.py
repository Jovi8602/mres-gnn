import os
import glob
import json
import pandas as pd

def main():
    metadata = (pd.read_csv('dataframes/metadata.csv').dropna(subset=["name_in_presence_absence"]))

    # Creating dictionary for {Strain:MDR}. Exports to JSON for use in GNN training.
    assemblies = glob.glob("assembled/*")
    mdr_dict = {}

    for file in assemblies:
        strain = os.path.basename(file).split(".")[0]
        mdr_status = metadata.loc[metadata['ID'] == strain, 'MDR'].tolist()
        if mdr_status:
            if mdr_status[0] == 'No':
                mdr_dict[strain] = 0
            elif mdr_status[0] == 'Yes':
                mdr_dict[strain] = 1

    print(len(mdr_dict), 'strains analysed for MDR.')
    print(mdr_dict)

    with open("dataframes/mdr1000.json", "w") as f:
        json.dump(mdr_dict, f)

if __name__ == "__main__":
    main()
