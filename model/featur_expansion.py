import pandas as pd
import numpy as np

# Load dataset
df = pd.read_excel("train_random_all_1_10.xlsx")

# Atomic composition features
df_feat = df.copy()
df_feat["C_frac"] = df_feat["C"] / 100.0
df_feat["H_frac"] = df_feat["H"] / 100.0
df_feat["O_frac"] = df_feat["O"] / 100.0
df_feat["N_frac"] = df_feat["N"] / 100.0

df_feat["DoU"] = (2 * df_feat["C"] + 2 + df_feat["N"] - df_feat["H"]) / 2
df_feat["O_enrichment"] = df_feat["O"] / (df_feat["C"] + 1e-6)
df_feat["C_O_interaction"] = df_feat["C"] * df_feat["O"]
df_feat["H_O_interaction"] = df_feat["H"] * df_feat["O"]

# Polymer functional group priors
polymer_priors = pd.DataFrame({
    "polymer": ["PET", "PE&PP", "PVC", "PS", "PA", "PC", "PBT", "PMMA", "PLA", "POM"],
    "Mw_monomer": [192.16, 28.05, 62.50, 104.15, 113.16, 254.28, 220.22, 100.12, 72.06, 30.03],
    "ester":     [1,0,0,0,0,0,1,1,1,0],
    "aromatic":  [1,0,0,1,0,1,1,0,0,0],
    "amide":     [0,0,0,0,1,0,0,0,0,0],
    "carbonate": [0,0,0,0,0,1,0,0,0,0]
})
polymer_priors.set_index("polymer", inplace=True)

# Weighted structural features for mixtures
target_labels = ["PET", "PE&PP", "PVC", "PS", "PA", "PC", "PBT", "PMMA", "PLA", "POM"]

def compute_weighted_structural_features(Y, polymer_priors, target_labels):
    Y_frac = Y / 100.0
    priors = polymer_priors.loc[target_labels]
    df_out = pd.DataFrame(index=range(Y.shape[0]))
    df_out["Mw_weighted"] = Y_frac @ priors["Mw_monomer"].values
    for fg in ["ester", "aromatic", "amide", "carbonate"]:
        df_out[f"{fg}_weighted"] = Y_frac @ priors[fg].values
    return df_out

# Generate extended dataset
Y_true = df[target_labels].values
df_struct = compute_weighted_structural_features(Y_true, polymer_priors, target_labels)
df_extended = pd.concat([df_feat, df_struct], axis=1)

# Export to Excel
df_extended.to_excel("extended_feature_dataset_01.xlsx", index=False)
print("Extended feature dataset saved as: extended_feature_dataset_01.xlsx")
