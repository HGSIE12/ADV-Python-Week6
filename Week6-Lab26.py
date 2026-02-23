import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 1) Load dataset
df = pd.read_csv("day26_distributions.csv")
numeric_df = df.select_dtypes(include=np.number)

# 2) Plot histogram + KDE for each numeric column
for i in numeric_df.columns:
    plt.figure(figsize=(6, 4))

    sns.histplot(numeric_df[i], bins=30, kde=True)

    mean_v = numeric_df[i].mean()
    median_v = numeric_df[i].median()

    plt.title(f"{i} Distribution")
    plt.xlabel(i)
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()

# 3) Compute summary statistics
de = pd.DataFrame({
    "mean": numeric_df.mean(),
    "median": numeric_df.median(),
    "std": numeric_df.std(),
    "skew": numeric_df.skew()
})

print("\nSummary Statistics:")
print(de)

# 4) Apply log1p to most skewed feature
skewed = de["skew"].abs().sort_values(ascending=False).index[0]
print("\nMost Skewed Feature:", skewed)

df[skewed + "_log"] = np.log1p(df[skewed])

# Replot before and after
plt.figure(figsize=(6, 4))
sns.kdeplot(df[skewed], label="Original", fill=True)
sns.kdeplot(df[skewed + "_log"], label="Log1p", fill=True)

plt.title(f"Effect of log1p on {skewed}")
plt.xlabel("skewed_feature")
plt.ylabel("Density")
plt.legend()
plt.show()
