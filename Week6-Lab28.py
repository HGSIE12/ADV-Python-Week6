import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#1) Load dataset
data = pd.read_csv("day28_groupby.csv")

#2) Compute groupby summaries
mul_group = data.groupby(["region", "segment"]).agg(
["mean", "median", "std", "count"]
)

print(mul_group.head(10))

#3) Plot aggregated metric
m = data.groupby("region")["sales"].agg("mean")
sns.barplot(x=m.index,y=m.values)
plt.title("Mean by Region")
plt.show()