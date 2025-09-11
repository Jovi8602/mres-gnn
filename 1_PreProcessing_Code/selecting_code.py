import pandas as pd

# Load + filter
metadata = (pd.read_csv('dataframes/metadata.csv')).dropna(subset=["name_in_presence_absence"])
df = metadata[(metadata["MDR"] != "?") & (metadata["ID"].str.startswith("ESC"))].copy()
df["strain"] = df["ID"].str[:-3]
df["PopPUNK"] = df["PopPUNK"].astype(int)

# Add first strain per lineage, then randomise the rest
seeded = (df[df["PopPUNK"].between(1, 51)]
          .drop_duplicates("PopPUNK", keep="first")["strain"].tolist())
remaining = df.loc[~df["strain"].isin(seeded), "strain"]
need = max(0, 1500)
extra = remaining.sample(n=need, random_state=42).tolist()
entero_list = seeded + extra

# Write to list
with open("esc_strains.txt", "w") as f:
    f.write(",".join(entero_list))

print(f"{len(entero_list)} strains prepared. Lineages included: {len(seeded)}.")