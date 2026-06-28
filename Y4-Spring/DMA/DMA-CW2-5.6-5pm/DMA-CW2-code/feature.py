import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import seaborn as sns

sns.set(style="whitegrid")


df_original = pd.read_csv("data.csv")
df = df_original.copy()
df_original.head()


df.shape
df.info()
df.describe()

null_counts = df.isnull().sum()
null_percent = (df.isnull().sum() / len(df)) * 100

missing_summary = pd.concat([null_counts, null_percent], axis=1, keys=['Total', 'Percentage'])
missing_summary = missing_summary[missing_summary['Total'] > 0].sort_values(by='Percentage', ascending=False)

print("Columns with missing values (Top 10):")
print(missing_summary.head(10))

if not missing_summary.empty:
    plt.figure(figsize=(12, 6))
    sns.heatmap(df[missing_summary.index].isnull(), cbar=False, cmap='viridis')
    plt.title("Heatmap of Missing Values (Filtered Features)")
    plt.xlabel("Features with Missing Data")
    plt.show()
else:
    print("No missing values detected.")


price_skew = df['price'].skew()
print(f"Skewness of Price: {price_skew:.2f}")

plt.figure(figsize=(8, 5))
sns.histplot(df['price'], bins=50, kde=True) 

plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))

plt.title(f"Distribution of House Prices (Skewness: {price_skew:.2f})")
plt.xlabel("Price")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)
plt.show()


correlation = df['sqft_living'].corr(df['price'])

plt.figure(figsize=(8, 5))
sns.regplot(x="sqft_living", y="price", data=df, 
            scatter_kws={"alpha":0.3, "color":"navy"}, 
            line_kws={"color":"red"}) 

plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))

plt.title(f"Living Area vs Price (Correlation: {correlation:.2f})")
plt.xlabel("Living Area (sqft)")
plt.ylabel("Price")
plt.grid(True, linestyle='--', alpha=0.6)
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


plt.figure(figsize=(8, 5))

sns.regplot(
    x="sqft_above", 
    y="price", 
    data=df,
    scatter_kws={"alpha":0.3, "color":"navy"},
    line_kws={"color":"red"}
)

plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))

plt.title("Above-Ground Living Area vs Price")
plt.xlabel("Sqft Above Ground")
plt.ylabel("Price")
plt.grid(True, linestyle='--', alpha=0.5)

plt.show()

plt.figure(figsize=(12, 6))

sns.boxplot(
    x="bedrooms", 
    y="price", 
    data=df, 
    hue="bedrooms", 
    palette="magma", 
    legend=False
)

plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))

plt.title("Relationship between Number of Bedrooms and House Prices")
plt.xlabel("Number of Bedrooms")
plt.ylabel("Price")
plt.grid(axis='y', linestyle='--', alpha=0.3)

plt.show()


plt.figure(figsize=(8, 5))

sns.boxplot(
    x="bathrooms", 
    y="price", 
    hue="bathrooms",     
    data=df,
    palette="coolwarm",
    legend=False        
)

sns.pointplot(
    x="bathrooms",
    y="price",
    data=df,
    color="black",
    markers="D"
)

plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))

plt.title("Bathrooms vs House Price")
plt.xlabel("Number of Bathrooms")
plt.ylabel("Price")
plt.grid(axis='y', linestyle='--', alpha=0.4)

plt.show()

df["above_ratio"] = df["sqft_above"] / df["sqft_living"]

plt.figure(figsize=(8, 5))

sns.regplot(
    x="above_ratio",
    y="price",
    data=df,
    scatter_kws={"alpha":0.3},
    line_kws={"color":"red"}
)

plt.title("Above Ratio vs Price")
plt.xlabel("Above / Living Ratio")
plt.ylabel("Price")

plt.show()


df["room_density"] = df["bedrooms"] / df["sqft_living"]

plt.figure(figsize=(8, 5))

sns.regplot(
    x="room_density",
    y="price",
    data=df,
    scatter_kws={"alpha":0.3},
    line_kws={"color":"purple"}
)

plt.title("Room Density vs Price")
plt.xlabel("Bedrooms per Sqft")
plt.ylabel("Price")

plt.show()


features_to_add = ["above_ratio", "room_density"]

df_model = pd.get_dummies(df, drop_first=True)

X = df_model.drop(columns=["price"])


df["basement_ratio"] = df["sqft_basement"] / df["sqft_living"]
plt.figure(figsize=(8, 5))

sns.regplot(
    x="basement_ratio",
    y="price",
    data=df,
    scatter_kws={"alpha": 0.3},
    line_kws={"color": "red"}
)

plt.title("Basement Ratio vs House Price")
plt.xlabel("Basement / Living Ratio")
plt.ylabel("Price")

plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))

plt.grid(True, linestyle='--', alpha=0.4)

plt.show()

df["bathroom_density"] = df["bathrooms"] / df["sqft_living"]
plt.figure(figsize=(8, 5))

sns.regplot(
    x="bathroom_density",
    y="price",
    data=df,
    scatter_kws={"alpha": 0.3},
    line_kws={"color": "red"}
)

plt.title("Bathroom Density vs House Price")
plt.xlabel("Bathrooms / Living Area")
plt.ylabel("Price")

plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))

plt.grid(True, linestyle='--', alpha=0.4)

plt.show()

df["bath_bed_ratio"] = df["bathrooms"] / (df["bedrooms"] + 1e-6)
df["bath_bed_bin"] = pd.cut(df["bath_bed_ratio"], bins=5)

plt.figure(figsize=(8, 5))

sns.boxplot(
    x="bath_bed_bin",
    y="price",
    data=df
)

plt.title("Bathroom-to-Bedroom Ratio (Binned) vs Price")
plt.xlabel("Ratio (Binned)")
plt.ylabel("Price")

plt.xticks(rotation=30)
plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))

plt.show()


df["total_sqft"] = df["sqft_living"] + df["sqft_lot"]
plt.figure(figsize=(8, 5))

sns.regplot(
    x="total_sqft",
    y="price",
    data=df,
    scatter_kws={"alpha": 0.3},
    line_kws={"color": "red"}
)

plt.title("Total Property Size vs House Price")
plt.xlabel("Total Square Footage (Living + Lot)")
plt.ylabel("Price")

plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))

plt.grid(True, linestyle='--', alpha=0.4)

plt.show()


cols_to_drop = ["street", "country", "date"]

df_cleaned = df.drop(columns=cols_to_drop, errors="ignore")

print(f"Original features count: {df.shape[1]}")
print(f"Cleaned features count: {df_cleaned.shape[1]}")
print(f"Remaining columns: {df_cleaned.columns.tolist()}")

df_cleaned.head()

original_shape = df.shape[0]

Q1 = df["price"].quantile(0.25)
Q3 = df["price"].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df_cleaned = df[(df["price"] >= lower_bound) & (df["price"] <= upper_bound)].copy()

removed = original_shape - df_cleaned.shape[0]

print(f"Original data size: {original_shape}")
print(f"After removing outliers: {df_cleaned.shape[0]}")
print(f"Number of removed outliers: {removed}")
print(f"Percentage removed: {removed / original_shape * 100:.2f}%")

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

original_price = df_log["price"].copy()

df_log["log_price"] = np.log1p(df_log["price"])

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
sns.histplot(original_price, bins=30, kde=True, color='skyblue') 
plt.title("Original Price Distribution")

plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}')) 
plt.xlabel("Price")
plt.xticks(rotation=45) 

plt.subplot(1,2,2)
sns.histplot(df_log["log_price"], bins=30, kde=True, color='salmon')
plt.title("Log-Transformed Price Distribution")
plt.xlabel("Log(Price)")

plt.tight_layout()
plt.show()
print("Skewness before:", original_price.skew())
print("Skewness after:", df_log["log_price"].skew())


df_cleaned["date"] = pd.to_datetime(df_cleaned["date"])

df_cleaned["year"] = df_cleaned["date"].dt.year
df_cleaned["month"] = df_cleaned["date"].dt.month
print(df_cleaned[["year", "month"]].head())

df_cleaned["house_age"] = df_cleaned["year"] - df_cleaned["yr_built"]

# print(df_cleaned[df_cleaned["house_age"] < 0]) 

plt.figure(figsize=(10, 6))

sns.regplot(
    x="house_age",
    y="price",
    data=df_cleaned,
    scatter_kws={'alpha':0.2, 's':10}, 
    line_kws={'color':'red'}          
)

plt.gca().yaxis.set_major_formatter(StrMethodFormatter('${x:,.0f}'))

plt.title("Impact of House Age on Sale Price")
plt.xlabel("House Age (Years)")
plt.ylabel("Price (USD)")
plt.grid(True, linestyle='--', alpha=0.5)

plt.show()

df_cleaned["is_renovated"] = df_cleaned["yr_renovated"].apply(lambda x: 0 if x == 0 else 1)

plt.figure(figsize=(8, 5))

sns.barplot(x="is_renovated", y="price", data=df_cleaned, hue="is_renovated", palette="Set1", legend=False)

plt.gca().yaxis.set_major_formatter(StrMethodFormatter('${x:,.0f}'))

plt.title("Average House Price: Renovated vs Non-Renovated")
plt.xlabel("Has Been Renovated? (0 = No, 1 = Yes)")
plt.ylabel("Average Price (USD)")
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.show()


def get_season(month):
    if month in [12, 1, 2]: return "winter"
    elif month in [3, 4, 5]: return "spring"
    elif month in [6, 7, 8]: return "summer"
    else: return "autumn"

df_cleaned["season"] = df_cleaned["month"].apply(get_season)

season_order = ["spring", "summer", "autumn", "winter"]
seasonal_price = df_cleaned.groupby("season")["price"].mean().reindex(season_order)

plt.figure(figsize=(8, 5))
seasonal_price.plot(kind="bar", color='teal')

plt.gca().yaxis.set_major_formatter(StrMethodFormatter('${x:,.0f}'))

plt.title("Average House Price by Season")
plt.xlabel("Season")
plt.ylabel("Average Price (USD)")
plt.xticks(rotation=0) 
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

df_cleaned["log_total_sqft"] = np.log1p(df_cleaned["total_sqft"])

plt.figure(figsize=(10, 6))

sns.regplot(
    x="log_total_sqft",
    y="price",
    data=df_cleaned,
    scatter_kws={"alpha":0.3, "s":10},
    line_kws={"color":"red"}
)

plt.title("Log Total Sqft vs Price")
plt.xlabel("Log(Total Square Footage)")
plt.ylabel("Price (USD)")

plt.gca().yaxis.set_major_formatter(StrMethodFormatter('${x:,.0f}'))
plt.grid(True, linestyle='--', alpha=0.5)

plt.show()

df_cleaned["year_since_renovation"] = np.where(
    df_cleaned["yr_renovated"] == 0, 
    0, 
    df_cleaned["year"] - df_cleaned["yr_renovated"]
)

plt.figure(figsize=(10, 6))
sns.regplot(
    x="year_since_renovation",
    y="price",
    data=df_cleaned,
    scatter_kws={"alpha":0.2, "s":10},
    line_kws={"color":"orange"} 
)

plt.gca().yaxis.set_major_formatter(StrMethodFormatter('${x:,.0f}'))

plt.title("Recency of Renovation vs House Price")
plt.xlabel("Years Since Last Renovation (0 = Never or Just Renovated)")
plt.ylabel("Price")
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

print(f"Correlation (Year Since Renovation): {df_cleaned['year_since_renovation'].corr(df_cleaned['price']):.4f}")


# ===== Step 1：Feature Engineering（全部在这里做）=====

# 时间特征
df["date"] = pd.to_datetime(df["date"])
df["year"] = df["date"].dt.year

# 房龄
df["house_age"] = df["year"] - df["yr_built"]

# 翻新
df["is_renovated"] = df["yr_renovated"].apply(lambda x: 0 if x == 0 else 1)

# ratio特征
df["above_ratio"] = df["sqft_above"] / (df["sqft_living"] + 1e-6)
df["room_density"] = df["bedrooms"] / (df["sqft_living"] + 1e-6)
df["basement_ratio"] = df["sqft_basement"] / (df["sqft_living"] + 1e-6)
df["bathroom_density"] = df["bathrooms"] / (df["sqft_living"] + 1e-6)
df["bath_bed_ratio"] = df["bathrooms"] / (df["bedrooms"] + 1e-6)
df["total_sqft"] = df["sqft_living"] + df["sqft_lot"]

# ===== Step 2：Cleaning =====
df_cleaned = df.drop(columns=['street', 'country', 'date'], errors='ignore')

# ===== Step 3：Encoding =====
df_model = pd.get_dummies(df_cleaned, drop_first=True)

# ===== Step 4：Correlation =====
corr = df_model.corr(numeric_only=True)
corr_price = corr["price"].abs().sort_values(ascending=False)

# ===== Step 5：Feature selection =====
selected_features = [
    f for f in corr_price.index
    if f != "price" and corr_price[f] > 0.2
]

# ⭐ 手动补充（关键）
important_features = ["waterfront", "view", "is_renovated", "house_age"]

selected_features = list(set(selected_features + important_features))

print("Final selected features:")
print(selected_features)

# ===== Step 6：检查 =====
for f in important_features:
    print(f"{f}:", corr_price[f] if f in corr_price else "Not found")


corr_selected = df_model[selected_features].corr().abs()

upper = corr_selected.where(np.triu(np.ones(corr_selected.shape), k=1).astype(bool))

to_drop = [col for col in upper.columns if any(upper[col] > 0.8)]

print("Highly correlated features to drop:", to_drop)

final_features = [f for f in selected_features if f not in to_drop]
print("Final selected features:")
print(final_features)

corr_matrix = df_model[final_features].corr()

mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

plt.figure(figsize=(10, 8))

sns.heatmap(
    corr_matrix,
    cmap="Greys",      
    cbar=False,
    square=True
)

sns.heatmap(
    corr_matrix,
    mask=mask,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    square=True,
    linewidths=0.5,
    cbar_kws={"shrink": 0.8}
)

plt.title("Correlation Matrix of Housing Features")
plt.show()


plt.figure(figsize=(6, 8))

corr_price[final_features].sort_values().plot(kind="barh")

plt.title("Feature Correlation with Price")
plt.xlabel("Correlation")
plt.show()


