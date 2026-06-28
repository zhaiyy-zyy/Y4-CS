import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score

from xgboost import XGBRegressor
import shap

sns.set(style="whitegrid")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 120)

df_original = pd.read_csv("data.csv")
df = df_original.copy()

print("Dataset shape:", df.shape)
display(df.head())
display(df.info())
display(df.describe())

null_counts = df.isnull().sum()
null_percent = (df.isnull().sum() / len(df)) * 100

missing_summary = pd.concat(
    [null_counts, null_percent],
    axis=1,
    keys=["Total", "Percentage"]
)

missing_summary = missing_summary[missing_summary["Total"] > 0].sort_values(
    by="Percentage", ascending=False
)

print("Columns with missing values:")
display(missing_summary)

if not missing_summary.empty:
    plt.figure(figsize=(12, 6))
    sns.heatmap(df[missing_summary.index].isnull(), cbar=False, cmap="viridis")
    plt.title("Heatmap of Missing Values")
    plt.xlabel("Features with Missing Data")
    plt.show()
else:
    print("No missing values detected.")


price_skew = df["price"].skew()
print(f"Skewness of Price: {price_skew:.2f}")

plt.figure(figsize=(8, 5))
sns.histplot(df["price"], bins=50, kde=True)

plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
plt.title(f"Distribution of House Prices (Skewness: {price_skew:.2f})")
plt.xlabel("Price")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.grid(axis="y", alpha=0.3)
plt.show()

correlation = df["sqft_living"].corr(df["price"])

plt.figure(figsize=(8, 5))
sns.regplot(
    x="sqft_living",
    y="price",
    data=df,
    scatter_kws={"alpha": 0.3, "color": "navy"},
    line_kws={"color": "red"}
)

plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
plt.title(f"Living Area vs Price (Correlation: {correlation:.2f})")
plt.xlabel("Living Area (sqft)")
plt.ylabel("Price")
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()



corr = df.corr(numeric_only=True)

mask_lower = np.tril(np.ones_like(corr, dtype=bool))

plt.figure(figsize=(12, 10))

sns.set_style("white")

sns.heatmap(
    corr,
    mask=mask_lower,
    cmap="Greys",
    cbar=False,
    square=True,
    linewidths=0,     
    alpha=0.25       
)

sns.heatmap(
    corr,
    mask=np.triu(np.ones_like(corr, dtype=bool)),
    annot=True,
    fmt=".2f",
    cmap='RdBu_r',
    center=0,
    square=True,
    linewidths=0,    
    cbar_kws={"shrink": .8}
)

plt.gca().set_facecolor('white')

plt.title("Correlation Matrix of Housing Features")
# plt.savefig("correlation_heatmap.png", bbox_inches='tight') 
plt.show()

city_order = df.groupby('city')['price'].mean().sort_values(ascending=False).index

plt.figure(figsize=(10, 12))
sns.barplot(
    x="price", 
    y="city", 
    data=df, 
    order=city_order, 
    hue="city",          
    palette="viridis", 
    legend=False,        
    estimator=np.mean, 
    errorbar=None
)

plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))

plt.title("Average House Price by City (Ranked)")
plt.xlabel("Average Price")
plt.ylabel("City")
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.show()

plt.figure(figsize=(8, 6))

sns.boxplot(
    x="view", 
    y="price", 
    data=df, 
    hue="view", 
    palette="viridis", 
    legend=False
)

plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))

plt.title("Relationship between View Rating and House Prices")
plt.xlabel("View Rating (0 = Poor, 4 = Excellent)")
plt.ylabel("Price")
plt.grid(axis='y', linestyle='--', alpha=0.4)

plt.show()

plt.figure(figsize=(8, 6))

sns.boxplot(
    x="waterfront", 
    y="price", 
    data=df, 
    hue="waterfront", 
    palette="Set2", 
    legend=False
)

plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))

plt.title("Impact of Waterfront View on House Prices")
plt.xlabel("Waterfront View (0: No, 1: Yes)")
plt.ylabel("Price")
plt.grid(axis='y', linestyle='--', alpha=0.4)

plt.show()

original_size = df.shape[0]

Q1 = df["price"].quantile(0.25)
Q3 = df["price"].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df_cleaned = df[
    (df["price"] >= lower_bound) &
    (df["price"] <= upper_bound)
].copy()

removed = original_size - df_cleaned.shape[0]
removed_pct = removed / original_size * 100

print("Original data size:", original_size)
print("After removing outliers:", df_cleaned.shape[0])
print("Number of removed outliers:", removed)
print(f"Percentage removed: {removed_pct:.2f}%")

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.boxplot(y=df["price"])
plt.title("Before IQR Filtering")

plt.subplot(1, 2, 2)
sns.boxplot(y=df_cleaned["price"])
plt.title("After IQR Filtering")

plt.tight_layout()
plt.show()


df_log = df_cleaned.copy()
df_log["log_price"] = np.log1p(df_log["price"])

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.histplot(df_log["price"], bins=30, kde=True, color="skyblue")
plt.title("Original Price Distribution")
plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
plt.xlabel("Price")
plt.xticks(rotation=45)

plt.subplot(1, 2, 2)
sns.histplot(df_log["log_price"], bins=30, kde=True, color="salmon")
plt.title("Log-Transformed Price Distribution")
plt.xlabel("Log(Price)")

plt.tight_layout()
plt.show()

print("Skewness before:", df_log["price"].skew())
print("Skewness after:", df_log["log_price"].skew())


df_fe = df_cleaned.copy()

df_fe["date"] = pd.to_datetime(df_fe["date"])

df_fe["year"] = df_fe["date"].dt.year
df_fe["month"] = df_fe["date"].dt.month

print(df_fe[["year", "month"]].head())

df_fe["house_age"] = df_fe["year"] - df_fe["yr_built"]

df_fe["is_renovated"] = (df_fe["yr_renovated"] > 0).astype(int)

df_fe["year_since_renovation"] = np.where(
    df_fe["yr_renovated"] == 0,
    0,
    df_fe["year"] - df_fe["yr_renovated"]
)

df_fe[["house_age", "is_renovated", "year_since_renovation"]].head()

season_map = {
    12: "winter", 1: "winter", 2: "winter",
    3: "spring", 4: "spring", 5: "spring",
    6: "summer", 7: "summer", 8: "summer",
    9: "autumn", 10: "autumn", 11: "autumn"
}

df_fe["season"] = df_fe["month"].map(season_map)

df_fe["season"].value_counts()
df_fe["above_ratio"] = df_fe["sqft_above"] / (df_fe["sqft_living"] + 1e-6)
df_fe["room_density"] = df_fe["bedrooms"] / (df_fe["sqft_living"] + 1e-6)
df_fe["basement_ratio"] = df_fe["sqft_basement"] / (df_fe["sqft_living"] + 1e-6)
df_fe["bathroom_density"] = df_fe["bathrooms"] / (df_fe["sqft_living"] + 1e-6)
df_fe["bath_bed_ratio"] = df_fe["bathrooms"] / (df_fe["bedrooms"] + 1e-6)

df_fe[[
    "above_ratio",
    "room_density",
    "basement_ratio",
    "bathroom_density",
    "bath_bed_ratio"
]].head()

df_fe["total_sqft"] = df_fe["sqft_living"] + df_fe["sqft_lot"]

df_fe["log_total_sqft"] = np.log1p(df_fe["total_sqft"])

df_fe[["total_sqft", "log_total_sqft"]].head()


plt.figure(figsize=(10, 6))
sns.regplot(
    x="house_age",
    y="price",
    data=df_fe,
    scatter_kws={"alpha": 0.2, "s": 10},
    line_kws={"color": "red"}
)
plt.gca().yaxis.set_major_formatter(StrMethodFormatter('${x:,.0f}'))
plt.title("Impact of House Age on Sale Price")
plt.xlabel("House Age (Years)")
plt.ylabel("Price (USD)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()


plt.figure(figsize=(8, 5))
sns.barplot(
    x="is_renovated",
    y="price",
    data=df_fe,
    hue="is_renovated",
    palette="Set1",
    legend=False
)
plt.gca().yaxis.set_major_formatter(StrMethodFormatter('${x:,.0f}'))
plt.title("Average House Price: Renovated vs Non-Renovated")
plt.xlabel("Has Been Renovated? (0 = No, 1 = Yes)")
plt.ylabel("Average Price (USD)")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()


plt.figure(figsize=(10, 6))
sns.regplot(
    x="log_total_sqft",
    y="price",
    data=df_fe,
    scatter_kws={"alpha": 0.3, "s": 10},
    line_kws={"color": "red"}
)
plt.gca().yaxis.set_major_formatter(StrMethodFormatter('${x:,.0f}'))
plt.title("Log Total Sqft vs Price")
plt.xlabel("Log(Total Square Footage)")
plt.ylabel("Price (USD)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()


df_model = df_fe.copy()

df_model = df_model.drop(
    columns=["street", "country", "date"],
    errors="ignore"
)

print("Remaining columns:", df_model.columns.tolist())


df_model = pd.get_dummies(df_model, drop_first=True)

print("Shape after encoding:", df_model.shape)

df_model = df_model.apply(
    lambda col: col.astype(int) if col.dtype == 'bool' else col
)

non_numeric_cols = df_model.select_dtypes(exclude=[np.number]).columns.tolist()

print("Non-numeric columns:", non_numeric_cols)

corr = df_model.corr(numeric_only=True)

corr_price = corr["price"].abs().sort_values(ascending=False)

selected_features = [
    f for f in corr_price.index
    if f != "price" and corr_price[f] > 0.2
]

print("Selected features (corr > 0.2):")
print(selected_features)
      

important_features = ["waterfront", "view", "is_renovated", "house_age"]
important_features = [f for f in important_features if f in df_model.columns]

selected_features = list(set(selected_features + important_features))

print("After adding important features:")
print(selected_features)


corr_selected = df_model[selected_features].corr().abs()

upper = corr_selected.where(
    np.triu(np.ones(corr_selected.shape), k=1).astype(bool)
)

to_drop = [
    col for col in upper.columns
    if any(upper[col] > 0.8)
]

print("Highly correlated features to drop:")
print(to_drop)

final_features = [
    f for f in selected_features if f not in to_drop
]

print("Final selected features:")
print(final_features)


corr = df_model[final_features + ["price"]].corr()

mask_lower = np.tril(np.ones_like(corr, dtype=bool))

plt.figure(figsize=(10, 8))

sns.set_style("white")

sns.heatmap(
    corr,
    mask=mask_lower,
    cmap="Greys",
    cbar=False,
    square=True,
    linewidths=0,
    alpha=0.25
)

sns.heatmap(
    corr,
    mask=np.triu(np.ones_like(corr, dtype=bool)),
    annot=True,
    fmt=".2f",
    cmap="RdBu_r",
    center=0,
    square=True,
    linewidths=0,
    cbar_kws={"shrink": 0.8}
)

plt.gca().set_facecolor("white")

plt.title("Correlation Matrix of Selected Features")
plt.show()


plt.figure(figsize=(6, 8))
corr_price[final_features].sort_values().plot(kind="barh")
plt.title("Feature Correlation with Price")
plt.xlabel("Absolute Correlation")
plt.show()


