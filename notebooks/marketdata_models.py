import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

from sklearn.cluster import KMeans
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor

os.getcwd()

market_data = pd.read_csv("data/processed/allmarketdata_int.csv")

all_num = []

for year in range(10):
    for i in range(0 + year*13,12 + year*13):
        all_num.append(i)
all_num

# to create dataset
no_totals_df = market_data.iloc[all_num]
no_totals_df.columns
no_totals_df.month = no_totals_df.month.apply(int)
no_totals_df = no_totals_df.sort_values(by=["year", "month"], ascending = True)

no_totals_df.to_csv("data/processed/months_marketdata.csv")

# to load in dataset
no_totals_df = pd.read_csv("data/processed/months_marketdata.csv")
no_totals_df.reset_index(drop=True)
#retail
x = range(len(all_num))
y = no_totals_df['retail_total']/(sum(no_totals_df['retail_total'])/len(no_totals_df['retail_total']))
y1 = no_totals_df['books_hobbies']/(sum(no_totals_df['books_hobbies'])/len(no_totals_df['books_hobbies']))
y2 = no_totals_df['grocery']/(sum(no_totals_df['grocery'])/len(no_totals_df['grocery']))
y3= no_totals_df['fuel']/(sum(no_totals_df['fuel'])/len(no_totals_df['fuel']))


plt.plot(x, y, label = "Retail")
plt.plot(x, y1, label = "Books & Hobbies")
plt.plot(x, y2, label = "Groceries")
plt.plot(x, y3, label = "Fuel")
plt.xlabel("Months")
plt.xticks(range(1, len(all_num), 12), no_totals_df.year[::12])
plt.ylabel("Total 2012 - 2022 Spending Change")
plt.legend()
plt.show()


monthly_df = no_totals_df.groupby('month').sum()
xm = range(1, 13)
ym = monthly_df['retail_total']
plt.plot(xm, ym, label = "Retail")
#plt.plot(x, y1, label = "Food")
plt.xlabel("Months")
plt.xticks(range(1,13))
plt.ylabel("Total 2013-2022 Retail Spending Change in Thousands of Millions of $")
plt.legend()
plt.show()

no_totals_df.iloc[:5]
no_totals_df.sort_index()
no_totals_df.sort_values(by = "month")

X = df_cat.copy()
y = X.pop("Scholarship")

kmeans = KMeans(10, random_state=0)
X["Cluster"] = kmeans.fit_predict(X_scaled)
X["Cluster"] = X["Cluster"].astype("category") 
Xy = X.copy()
Xy["Cluster"] = Xy.Cluster.astype("category")
Xy["Scholarship"] = y
sns.relplot(
    x="value", y="Scholarship", hue="Cluster", col="variable",
    height=4, aspect=1, facet_kws={'sharex': False}, col_wrap=3,
    data=Xy.melt(
        value_vars=features, id_vars=["Scholarship", "Cluster"],
    ),
);