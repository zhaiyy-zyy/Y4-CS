# Basic libraries
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import seaborn as sns

# Settings
sns.set(style="whitegrid")

# Load dataset
df_original = pd.read_csv("data.csv")

# Create a working copy for processing
df = df_original.copy()

# Preview data
df_original.head()

# Shape of dataset
df.shape

# Data types
df.info()

# Statistical summary
df.describe()


# Missing Value Analysis ===

# 1. Calculate the missing values and their percentages.
null_counts = df.isnull().sum()
null_percent = (df.isnull().sum() / len(df)) * 100

# Show columns with missing values
missing_summary = pd.concat([null_counts, null_percent], axis=1, keys=['Total', 'Percentage'])
missing_summary = missing_summary[missing_summary['Total'] > 0].sort_values(by='Percentage', ascending=False)

print("Columns with missing values (Top 10):")
print(missing_summary.head(10))

# 2. visualization
if not missing_summary.empty:
    plt.figure(figsize=(12, 6))
    sns.heatmap(df[missing_summary.index].isnull(), cbar=False, cmap='viridis')
    plt.title("Heatmap of Missing Values (Filtered Features)")
    plt.xlabel("Features with Missing Data")
    plt.show()
else:
    print("No missing values detected.")


# === 4. Target Variable Analysis (Price) ===

# Calculate Skewness
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

# Living area vs price

# Pearson Correlation
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


# Compute correlation matrix
corr = df.corr(numeric_only=True)

mask = np.triu(np.ones_like(corr, dtype=bool))

plt.figure(figsize=(12, 10))
sns.heatmap(corr, 
            mask=mask, 
            annot=True, 
            fmt=".2f", 
            cmap='RdBu_r', 
            center=0,
            square=True, 
            linewidths=.5, 
            cbar_kws={"shrink": .8})

plt.title("Correlation Matrix of Housing Features (Triangular Mask)")
# plt.savefig("correlation_heatmap.png", bbox_inches='tight') 
plt.show()

# City vs Price
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

from matplotlib.ticker import StrMethodFormatter
plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))

plt.title("Average House Price by City (Ranked)")
plt.xlabel("Average Price")
plt.ylabel("City")
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.show()

# === 7.1 Deep Dive: Top 10 High-Value Cities ===

top_10_cities = df.groupby("city")["price"].mean().sort_values(ascending=False).head(10).reset_index()

plt.figure(figsize=(10, 6))

sns.barplot(
    x="city", 
    y="price", 
    data=top_10_cities, 
    hue="city", 
    palette="magma", 
    legend=False
)

plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))

# Average/Mean House Price
for i, val in enumerate(top_10_cities['price']):
    plt.text(i, val, f'${val:,.0f}', ha='center', va='bottom', fontsize=9)

plt.title("Top 10 Most Expensive Cities by Average Price")
plt.xlabel("City")
plt.ylabel("Average Price")
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.show()

# Waterfront vs Price
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

# View vs Price
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


# Bedrooms vs Price (filtered)

limit = 6
df_filtered = df[df['bedrooms'] <= limit]
removed_count = len(df) - len(df_filtered)
print(f"Removed {removed_count} outlier properties with >{limit} bedrooms.")

plt.figure(figsize=(10, 6))

sns.boxplot(
    x="bedrooms", 
    y="price", 
    data=df_filtered, 
    hue="bedrooms", 
    palette="viridis", 
    legend=False
)

plt.gca().yaxis.set_major_formatter(StrMethodFormatter('${x:,.0f}'))

plt.title(f"House Price Distribution by Bedrooms (Filtered: <= {limit} Bedrooms)")
plt.xlabel("Number of Bedrooms")
plt.ylabel("Price (USD)")
plt.grid(axis='y', linestyle='--', alpha=0.3)

plt.show()

# Drop irrelevant features
cols_to_drop = ["street", "country", "statezip", "date"]

df_cleaned = df.drop(columns=cols_to_drop, errors="ignore")

print(f"Original features count: {df.shape[1]}")
print(f"Cleaned features count: {df_cleaned.shape[1]}")
print(f"Remaining columns: {df_cleaned.columns.tolist()}")

df_cleaned.head()

# Save original size
original_shape = df.shape[0]

# Detect outliers using IQR
Q1 = df["price"].quantile(0.25)
Q3 = df["price"].quantile(0.75)
IQR = Q3 - Q1

# Define bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filter dataset
df_cleaned = df[(df["price"] >= lower_bound) & (df["price"] <= upper_bound)].copy()

# Calculate removed points
removed = original_shape - df_cleaned.shape[0]

# Print results
print(f"Original data size: {original_shape}")
print(f"After removing outliers: {df_cleaned.shape[0]}")
print(f"Number of removed outliers: {removed}")
print(f"Percentage removed: {removed / original_shape * 100:.2f}%")

# Comparison of the distribution before and after cleaning
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.boxplot(y=df["price"])
plt.title("Before IQR Filtering")

plt.subplot(1, 2, 2)
sns.boxplot(y=df_cleaned["price"])
plt.title("After IQR Filtering")

plt.tight_layout()
plt.show()

# Log Transformation Analysis 

# Use cleaned data
df_log = df_cleaned.copy()

# Save original price
original_price = df_log["price"].copy()

# Apply log transformation
df_log["log_price"] = np.log1p(df_log["price"])

# Plot comparison
plt.figure(figsize=(12,5))

# Before
plt.subplot(1,2,1)
sns.histplot(original_price, bins=30, kde=True, color='skyblue') 
plt.title("Original Price Distribution")

plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}')) 
plt.xlabel("Price")
plt.xticks(rotation=45) 

# After
plt.subplot(1,2,2)
sns.histplot(df_log["log_price"], bins=30, kde=True, color='salmon')
plt.title("Log-Transformed Price Distribution")
plt.xlabel("Log(Price)")

plt.tight_layout()
plt.show()
print("Skewness before:", original_price.skew())
print("Skewness after:", df_log["log_price"].skew())

# Convert date to datetime
df_cleaned["date"] = pd.to_datetime(df_cleaned["date"])

# Extract year and month
df_cleaned["year"] = df_cleaned["date"].dt.year
df_cleaned["month"] = df_cleaned["date"].dt.month
print(df_cleaned[["year", "month"]].head())

# House age
df_cleaned["house_age"] = df_cleaned["year"] - df_cleaned["yr_built"]

# 2. 检查是否有逻辑错误（例如房龄为负数）
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


# Renovation indicator
df_cleaned["is_renovated"] = df_cleaned["yr_renovated"].apply(lambda x: 0 if x == 0 else 1)

plt.figure(figsize=(8, 5))

sns.barplot(x="is_renovated", y="price", data=df_cleaned, hue="is_renovated", palette="Set1", legend=False)

plt.gca().yaxis.set_major_formatter(StrMethodFormatter('${x:,.0f}'))

plt.title("Average House Price: Renovated vs Non-Renovated")
plt.xlabel("Has Been Renovated? (0 = No, 1 = Yes)")
plt.ylabel("Average Price (USD)")
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.show()

# Define season
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

# total square 
df_cleaned["total_sqft"] = df_cleaned["sqft_living"] + df_cleaned["sqft_lot"]

plt.figure(figsize=(10, 6))

sns.regplot(
    x="total_sqft",
    y="price",
    data=df_cleaned,
    scatter_kws={"alpha":0.2, "s":10},
    line_kws={"color":"red"}
)

plt.gca().yaxis.set_major_formatter(StrMethodFormatter('${x:,.0f}'))

plt.title("Total Property Size vs Price")
plt.xlabel("Total Square Footage (Living + Lot)")
plt.ylabel("Price (USD)")
plt.grid(True, linestyle='--', alpha=0.5)

plt.show()

corr_total = df_cleaned["total_sqft"].corr(df_cleaned["price"])
corr_living = df_cleaned["sqft_living"].corr(df_cleaned["price"])
print(f"Correlation (Total Sqft): {corr_total:.4f}")
print(f"Correlation (Living Sqft Only): {corr_living:.4f}")

# years since renovation
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

df_model = df_cleaned.copy()
df_model = df_model.drop(columns=['street', 'statezip', 'country', 'date'], errors='ignore')
df_model = pd.get_dummies(df_model, drop_first=True)

# Split features and target
X = df_model.drop(columns=["price", "log_price"], errors="ignore")
y = df_model["price"]

# Train-test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # 80% train, 20% test
    random_state=42
)

print("Train size:", X_train.shape)
print("Test size:", X_test.shape)



irrelevant_strings = ['street', 'statezip', 'country', 'date']

# 2. drop
df_model = df_model.drop(columns=irrelevant_strings, errors='ignore')

# 3. int (0/1)
df_model = df_model.apply(lambda x: x.astype(int) if x.dtype == 'bool' else x)

X = df_model.drop(columns=["price", "log_price"], errors="ignore")
y = df_model["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Remaining features:", X_train.columns.tolist())
print("Non-numeric count:", X_train.select_dtypes(exclude=[np.number]).shape[1])

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test = pd.DataFrame(X_test_scaled, columns=X.columns)

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score

# 5-Fold Out-of-Fold
# Level 1 - base model
base_models = {
    "Linear": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
    #"MLP": MLPRegressor(hidden_layer_sizes=(100,50), max_iter=2000, alpha=0.001, random_state=42)
    "MLP": MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=2000, alpha=0.001, learning_rate_init=0.001, early_stopping=True, n_iter_no_change=20, random_state=42)
}

results = []

for name, model in base_models.items():
    print(f"Training {name}...")
    
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    
    results.append([name, rmse, r2])

results_df = pd.DataFrame(results, columns=["Model", "RMSE", "R2"])

print("\n=== Base Model Performance ===")
print(results_df)

# initial
train_oof = np.zeros((X_train.shape[0], len(base_models)))
test_preds = np.zeros((X_test.shape[0], len(base_models)))

# set 5-Fold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# train
for i, (name, model) in enumerate(base_models.items()):
    print(f"Generating OOF for {name}...")
    
    # mean
    fold_test_preds = np.zeros((X_test.shape[0], 5))
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
        # vilidation
        X_f_train, X_f_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_f_train, y_f_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        model.fit(X_f_train, y_f_train)
        
        train_oof[val_idx, i] = model.predict(X_f_val)
        
        fold_test_preds[:, fold] = model.predict(X_test)
    
    test_preds[:, i] = fold_test_preds.mean(axis=1)


# 1. Meta-features
X_meta_train = pd.DataFrame(train_oof, columns=base_models.keys())
X_meta_test = pd.DataFrame(test_preds, columns=base_models.keys())

# 2.Level 2 
meta_learner = LinearRegression()
meta_learner.fit(X_meta_train, y_train)

y_final_pred = meta_learner.predict(X_meta_test)

print("Level 2: Meta-learner training successful.")
print("New Meta-features sample (Level 1 outputs):\n", X_meta_train.head())


from sklearn.metrics import mean_squared_error, r2_score
from IPython.display import display, Markdown

final_rmse = np.sqrt(mean_squared_error(y_test, y_final_pred))
final_r2 = r2_score(y_test, y_final_pred)

display(Markdown("---"))
display(Markdown("#### **Stacking Ensemble Evaluation Results**"))

results_df = pd.DataFrame({
    "Metric": ["Root Mean Squared Error (RMSE)", "R-squared Score ($R^2$)"],
    "Score": [f"${final_rmse:,.2f}", f"{final_r2:.4f}"]
})

display(results_df)

display(Markdown(f"""
> **Discussion:**
> The final model achieved an $R^2$ of **{final_r2:.4f}**. This indicates that our ensemble can explain approximately **{final_r2*100:.1f}%** of the variance in house prices. 
> Compared to the base models, the stacking approach effectively reduced the RMSE by leveraging the strengths of different algorithmic paradigms.
"""))


!pip install shap
# 快速查看随机森林的特征重要性

# 重新训练一个简单的 RF 来获取重要性（因为 Stacking 封装了模型）
rf_temp = RandomForestRegressor(n_estimators=100, random_state=42)
rf_temp.fit(X_train, y_train)

import shap

# 创建 explainer
explainer = shap.Explainer(rf_temp, X_train)

# 计算 SHAP values（用 test set）
shap_values = explainer(X_test, check_additivity=False)

# Summary plot（最重要）
shap.summary_plot(shap_values, X_test)

importances = pd.Series(rf_temp.feature_importances_, index=X_train.columns)
importances.nlargest(10).plot(kind='barh')
plt.title("Top 10 Most Influential Features")
plt.show()

# Hyperparameter Tunin
from sklearn.model_selection import GridSearchCV

param_grid = {
    "n_estimators": [100, 200, 300],
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [3, 5, 7],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0]
}

xgb = XGBRegressor(random_state=42)

grid = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    cv=5,
    scoring="neg_root_mean_squared_error",  
    n_jobs=-1,
    verbose=1
)

grid.fit(X_train, y_train)

best_xgb = grid.best_estimator_

print("Best Parameters:", grid.best_params_)
print("Best CV RMSE:", -grid.best_score_)

for name, model in base_models.items():
    print(f"Running CV for {name}...")
    
    try:
        scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=5,
            scoring="neg_root_mean_squared_error"
        )
        print(f"{name} CV RMSE:", -scores.mean())
    
    except Exception as e:
        print(f"{name} FAILED:", e)


results.append(["Stacking", final_rmse, final_r2])

results_df = pd.DataFrame(results, columns=["Model", "RMSE", "R2"])

print(results_df.sort_values(by="RMSE"))