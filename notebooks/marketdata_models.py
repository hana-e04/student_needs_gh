import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

from sklearn.cluster import KMeans
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


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
no_totals_df.reset_index(drop=True)
no_totals_cleaned_df = no_totals_df.iloc[:, 1:]
no_totals_cleaned_df.to_csv("data/processed/months_marketdata.csv")
no_totals_cleaned_df
# to load in dataset
load_in_df = pd.read_csv("data/processed/months_marketdata.csv")
no_totals_df = load_in_df.iloc[:, 1:]

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

no_totals_df.columns

X = no_totals_df
y = no_totals_df.pop("books_hobbies")

features = ['year', 'month', 'retail_total',
            'food_stores', 'hp_care', 'fuel', 'clothes']

# Standardize
X_scaled = X.loc[:, features]

kmeans = KMeans(10, random_state=0)
X["Cluster"] = kmeans.fit_predict(X_scaled)
X["Cluster"] = X["Cluster"].astype("float")
Xy = X.copy()
Xy["Cluster"] = Xy.Cluster.astype("float")
Xy["books_hobbies"] = y
sns.relplot(
    x="value", y="books_hobbies", hue="Cluster", col="variable",
    height=4, aspect=1, facet_kws={'sharex': False}, col_wrap=3,
    data=Xy.melt(
        value_vars=features, id_vars=["books_hobbies", "Cluster"],
    ),
)
plt.show()

# decision tree model
full_df = no_totals_df.copy()

X_full = full_df
y_full = full_df.pop("books_hobbies")

train_X, val_X, train_y, val_y = train_test_split(X_full, y_full, random_state = 1)

market_tree_model_full= DecisionTreeRegressor(random_state = 1)
market_tree_model_full.fit(train_X, train_y)

market_tree_predictions_full = market_tree_model_full.predict(val_X)

market_tree_predictions_full
val_y
val_mse_full = mean_squared_error(val_y, market_tree_predictions_full)
val_mse_full

val_mae_full = mean_absolute_error(val_y, market_tree_predictions_full)
val_mae_full

errors_full = np.array(val_y) - market_tree_predictions_full
market_tree_predictions_full

fig = plt.figure(figsize =(10, 7))

plt.boxplot(errors_full)
plt.show()

# decision tree model - FEATURE TESTING
X1 = no_totals_df.drop(columns=["year", "month", "Cluster"], axis=1)
y1 = X1.pop("books_hobbies")
X1.columns
train_X, val_X, train_y, val_y = train_test_split(X1, y1, random_state = 1)

market_tree_model= DecisionTreeRegressor(random_state = 1)
market_tree_model.fit(train_X, train_y)

market_tree_predictions = market_tree_model.predict(val_X)

market_tree_predictions
val_y
val_mse = mean_squared_error(val_y, market_tree_predictions)
val_mse

val_mae = mean_absolute_error(val_y, market_tree_predictions)
val_mae

errors_full = np.array(val_y) - market_tree_predictions
market_tree_predictions


plt.boxplot(errors)
plt.show()


# FEATURE TESTING
feature_test_df = no_totals_df.copy()

model_errors = {'col':["control"],
                'mae':[val_mae_full],
                'mse':[val_mse_full],
                'error':[errors_full]}
no_totals_df.columns

for col in feature_test_df.drop(columns="books_hobbies", axis=1).columns:
    X = feature_test_df.drop(columns=col, axis=1)
    y = X.pop("books_hobbies")
    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 1)

    market_tree_model= DecisionTreeRegressor(random_state = 1)
    market_tree_model.fit(train_X, train_y)

    market_tree_predictions = market_tree_model.predict(val_X)

    market_tree_predictions
    val_mse = mean_squared_error(val_y, market_tree_predictions)

    val_mae = mean_absolute_error(val_y, market_tree_predictions)
    errors = np.array(val_y) - market_tree_predictions

    model_errors['col'].append(col)
    model_errors['mae'].append(val_mae)
    model_errors['mse'].append(val_mse)
    model_errors['error'].append(errors)

model_errors['col']

# mean absolute error visualized
plt.bar(model_errors['col'], model_errors['mae'])
plt.title('MAE by column excluded')
plt.xlabel('Column')
plt.ylabel('Mean Absolute Error')
plt.axhline(y=model_errors['mae'][0], color='r', linestyle='--', label='Horizontal Line')
plt.show()

# mean squared error visualized
plt.bar(model_errors['col'], model_errors['mse'])
plt.title('MSE by column excluded')
plt.xlabel('Column')
plt.ylabel('Mean Squared Error')
plt.axhline(y=model_errors['mse'][0], color='r', linestyle='--', label='Horizontal Line')
plt.show()

