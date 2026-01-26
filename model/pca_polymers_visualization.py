import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# ------------------------
# 1. Dataset: elemental composition of plastics
# ------------------------
data = {
    'Plastic': ['PE','PP','PVC','PS','PET','PBT','POM','PMMA','PLA','PA','PC'],
    'C': [85.63,85.63,38.44,92.26,62.5,65.45,40.0,59.98,50.0,63.69,75.58],
    'H': [14.37,14.37,4.84,7.74,4.2,5.49,6.71,8.05,5.6,9.8,5.55],
    'O': [0,0,0,0,33.3,29.06,53.28,31.96,44.4,14.14,18.87],
    'N': [0,0,0,0,0,0,0,0,0,12.38,0],
    'H/C': [2,2,1.5,1,0.8,1,2,1.6,1.333,1.833,0.875],
    'O/C': [0,0,0,0,0.4,0.333,1,0.4,0.667,0.167,0.1875]
}
df = pd.DataFrame(data)

# ------------------------
# 2. PCA (2 components)
# ------------------------
features = ['C','H','O','N','H/C','O/C']
pca = PCA(n_components=2)
pcs = pca.fit_transform(df[features])
df['PC1'], df['PC2'] = pcs[:,0], pcs[:,1]

# ------------------------
# 3. Color and marker settings
# ------------------------
custom_palette = {
    'PE': '#1f77b4', 'PP': '#1f77b4', 'PVC': '#2ca02c', 'PS': '#9467bd',
    'PET': '#ff6666', 'PBT': '#d62728', 'PMMA': '#ff9999', 'POM': '#7f7f7f',
    'PLA': '#bcbd22', 'PA': '#8c564b', 'PC': '#e377c2'
}

labels_with_formula = {
    'PE': 'PE\u2003(C₂H₄)ₙ', 'PP': 'PP\u2003(C₃H₆)ₙ', 'PVC': 'PVC\u2002(C₂H₃Cl)ₙ',
    'PS': 'PS\u2003(C₈H₈)ₙ', 'PET': 'PET\u2001(C₁₀H₈O₄)ₙ', 'PA': 'PA\u2003(C₆H₁₁NO)ₙ',
    'PC': 'PC\u2003(C₁₆H₁₄O₃)ₙ', 'PBT': 'PBT\u2003(C₁₂H₁₂O₄)ₙ',
    'POM': 'POM\u2003(CH₂O)ₙ', 'PMMA': 'PMMA\u2003(C₅O₂H₈)ₙ', 'PLA': 'PLA\u2003(C₃H₄O₂)ₙ'
}

extended_set = ['PBT', 'POM', 'PMMA', 'PLA']  # Mark extended plastics as squares

# ------------------------
# 4. Plot PCA
# ------------------------
plt.figure(figsize=(8,5), dpi=700)
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 18

for plastic in df['Plastic']:
    subset = df[df['Plastic'] == plastic]
    marker_style = 's' if plastic in extended_set else 'o'
    color = custom_palette[plastic]

    # Special handling for PE & PP
    if plastic == 'PE':
        plt.scatter(subset['PC1'], subset['PC2'],
                    s=180, color=color, edgecolor='black', alpha=0.9,
                    marker=marker_style, label=labels_with_formula[plastic])
    elif plastic == 'PP':
        plt.scatter(subset['PC1'], subset['PC2'],
                    s=180, facecolors='none', edgecolors=color, linewidths=2,
                    alpha=0.9, marker=marker_style, label=labels_with_formula[plastic])
    else:
        plt.scatter(subset['PC1'], subset['PC2'],
                    s=180, color=color, edgecolor='black', alpha=0.9,
                    marker=marker_style, label=labels_with_formula[plastic])

# Adjust legend order
order = ['PE','PP','PVC','PS','PA','PC','PET','PBT','PMMA','POM','PLA']
handles, labels = plt.gca().get_legend_handles_labels()
order_idx = [labels.index(labels_with_formula[o]) for o in order]
plt.legend([handles[i] for i in order_idx],
           [labels[i] for i in order_idx],
           bbox_to_anchor=(1.02, 0.5), loc='center left',
           fontsize=14, frameon=False, handletextpad=0.5)

# Axes
plt.tick_params(axis='both', which='major', labelsize=18, width=1)
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_linewidth(1)

plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)", fontsize=18, fontweight='bold')
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)", fontsize=18, fontweight='bold')

plt.tight_layout()
plt.savefig("PCA_polymers_grouped_hollow_PP.png", dpi=700, transparent=True, bbox_inches='tight')
plt.show()
