import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
wine_quality = fetch_ucirepo(id=186)
# variable information
print(wine_quality.variables)

#data exploration
df = pd.concat([wine_quality.data.features, wine_quality.data.targets], axis=1)
print(df.isnull().sum())
print(df.describe())
print(df.corr())
##heatmap PLOT
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.show()
##target distribution PLOT
sns.countplot(x="quality", data=df)
plt.show()
