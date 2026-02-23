import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1) Load regression dataset
df = pd.read_csv("day26_distributions.csv")
print("Original Data:")
print(df)

#2) Plot histograms and KDEs

plt.figure(figsize=(6, 4))
df["income"].hist(bins=30, edgecolor="black")
plt.title("Income Distribution — Histogram")
plt.xlabel("Income"); plt.ylabel("Count")
plt.tight_layout(); plt.show()

plt.figure(figsize=(6, 4))
sns.kdeplot(df["income"], fill=True)
plt.title("Income Distribution — KDE")
plt.xlabel("Income"); plt.ylabel("Density")
plt.tight_layout(); plt.show()

#3) Compute mean, median, std, skew
numeric_df = df.select_dtypes(include=np.number)
stats = pd.DataFrame({
    "mean": numeric_df.mean(),
    "median": numeric_df.median(),
    "std": numeric_df.std(),
    "skew": numeric_df.skew()
})

print("Summary Statistics:")
print(stats)

# 4) Apply log1p to most skewed feature
skewed_feature = stats["skew"].abs().sort_values(ascending=False).index[0]
print("Most Skewed Feature:", skewed_feature)

df[skewed_feature + "_log"] = np.log1p(df[skewed_feature])

# Replot after transformation
plt.figure(figsize=(6,4))
sns.kdeplot(df[skewed_feature], label="Original", fill=True)
sns.kdeplot(df[skewed_feature + "_log"], label="Log1p", fill=True)
plt.legend()
plt.title(f"Before vs After log1p ({skewed_feature})")
plt.show()
