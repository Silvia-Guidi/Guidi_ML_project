import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from ucimlrepo import fetch_ucirepo
wine_quality = fetch_ucirepo(id=186)
# variable information
print(wine_quality.variables)
#print(wine_quality.metadata) 

#data exploration
df = pd.concat([wine_quality.data.features, wine_quality.data.targets], axis=1)
#print(df.isnull().sum())

fig, axes = plt.subplots(3, 4, figsize=(16, 10))  
axes = axes.flatten()
for i, col in enumerate(df.columns):  
    sns.histplot(df[col], kde=False, ax=axes[i], color="#2CEAA3")  
    axes[i].set_title(f"Distribution of {col}", fontsize=12)
fig.suptitle("Distribution of the variables: histograms", fontsize=16)
plt.tight_layout()
#plt.show()

fig, axes = plt.subplots(3, 4, figsize=(16,10))  # 3 rows, 4 columns
axes = axes.flatten()

for i, col in enumerate(df.columns):
    axes[i].boxplot(df[col], patch_artist=True, boxprops=dict(facecolor="#2CEAA3"))
    axes[i].set_title(col)
fig.suptitle("Distribition of the variables: boxplots", fontsize=16)
plt.tight_layout()
#plt.show()

#print(df.describe())
#print(df.corr())

##heatmap PLOT
plt.figure(figsize=(8,15))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Correlation Heatmap", fontsize=16, pad=20)
plt.show()

