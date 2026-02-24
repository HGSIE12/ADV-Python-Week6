import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("day27_boxplots.csv")


df["score"].hist(bins=10)
plt.title("Histogram for score")
plt.show()


plt.figure(figsize=(6, 4))
plt.boxplot(df["score"], vert=True, showfliers=True)
plt.title("Score — Boxplot"); plt.ylabel("Score")
plt.show()
