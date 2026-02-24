import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("day29_correlations.csv")

#1) Compute correlation matrix
c_matrix = df[["feature_x", "feature_y", "feature_z","target"]].corr()
print(c_matrix)

#2) Plot heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(c_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

#3) Plot scatter plots for strong/weak pairs
plt.figure(figsize=(6, 4))
plt.scatter(df["feature_x"], df["feature_y"], alpha=0.5)
plt.title("feature_x vs feature_y")
plt.xlabel("feature_x")
plt.ylabel("feature_y")
plt.show()