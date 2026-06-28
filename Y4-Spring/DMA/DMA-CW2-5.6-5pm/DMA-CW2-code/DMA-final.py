import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone

from sklearn.linear_model import Ridge, LassoCV
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from xgboost import XGBRegressor

import warnings
warnings.filterwarnings('ignore')
sns.set(style='whitegrid')

df_original = pd.read_csv('data.csv')
df = df_original.copy()
print('Shape:', df.shape)
display(df.head())
df.info()
display(df.describe())


null_counts = df.isnull().sum()
null_pct = (df.isnull().sum() / len(df)) * 100
missing = pd.concat([null_counts, null_pct], axis=1, keys=['Count', 'Pct'])
missing = missing[missing['Count'] > 0].sort_values('Pct', ascending=False)
print('Missing values:')
display(missing if len(missing) > 0 else 'No missing values')

price_skew = df['price'].skew()
plt.figure(figsize=(8, 5))
sns.histplot(df['price'], bins=50, kde=True)
plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
plt.title(f'Distribution of House Prices (Skewness: {price_skew:.2f})')
plt.xlabel('Price')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.regplot(x='sqft_living', y='price', data=df, scatter_kws={'alpha': 0.3}, ax=axes[0])
axes[0].set_title(f'Living Area vs Price (r={df["sqft_living"].corr(df["price"]):.2f})')

sns.boxplot(x='waterfront', y='price', data=df, hue='waterfront', palette='Set2', legend=False, ax=axes[1])
axes[1].set_title('Waterfront vs Price')

sns.boxplot(x='view', y='price', data=df, hue='view', palette='viridis', legend=False, ax=axes[2])
axes[2].set_title('View vs Price')

for ax in axes:
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))

plt.tight_layout()
plt.show()


corr = df.corr(numeric_only=True)
mask = np.triu(np.ones_like(corr, dtype=bool))

plt.figure(figsize=(12, 10))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0, square=True, cbar_kws={'shrink': .8})
plt.title('Correlation Matrix')
plt.show()


city_order = df.groupby('city')['price'].mean().sort_values(ascending=False).index

plt.figure(figsize=(10, 12))
sns.barplot(x='price', y='city', data=df, order=city_order, hue='city', palette='viridis', legend=False, estimator=np.mean, errorbar=None)
plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
plt.title('Average House Price by City')
plt.show()

original_size = df.shape[0]
df_cleaned = df[df['price'] < df['price'].quantile(0.99)].copy()
removed = original_size - df_cleaned.shape[0]
print(f'Removed {removed} outliers ({removed/original_size*100:.1f}%), {df_cleaned.shape[0]} rows remain')

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.boxplot(y=df['price'], ax=axes[0])
axes[0].set_title('Before')
sns.boxplot(y=df_cleaned['price'], ax=axes[1])
axes[1].set_title('After')
plt.tight_layout()
plt.show()


df_fe = df_cleaned.copy()

df_fe['date'] = pd.to_datetime(df_fe['date'])
df_fe['year'] = df_fe['date'].dt.year
df_fe['month'] = df_fe['date'].dt.month

df_fe['house_age'] = df_fe['year'] - df_fe['yr_built']
df_fe['is_renovated'] = (df_fe['yr_renovated'] > 0).astype(int)
df_fe['years_since_renovation'] = np.where(df_fe['yr_renovated'] > 0, df_fe['year'] - df_fe['yr_renovated'], 0)

EPS = 1e-6
df_fe['basement_ratio'] = df_fe['sqft_basement'] / (df_fe['sqft_living'] + EPS)
df_fe['has_basement'] = (df_fe['sqft_basement'] > 0).astype(int)
df_fe['bath_bed_ratio'] = df_fe['bathrooms'] / (df_fe['bedrooms'] + EPS)

season_map = {12:'winter',1:'winter',2:'winter',3:'spring',4:'spring',5:'spring',
              6:'summer',7:'summer',8:'summer',9:'autumn',10:'autumn',11:'autumn'}
df_fe['season'] = df_fe['month'].map(season_map)

print('Engineered features added')
print(df_fe[['house_age','is_renovated','years_since_renovation','basement_ratio','has_basement','bath_bed_ratio','season']].head())


fig, axes = plt.subplots(1, 3, figsize=(16, 5))

sns.regplot(x='house_age', y='price', data=df_fe, scatter_kws={'alpha': 0.2, 's': 10}, line_kws={'color': 'red'}, ax=axes[0])
axes[0].set_title('House Age vs Price')

sns.barplot(x='is_renovated', y='price', data=df_fe, hue='is_renovated', palette='Set1', legend=False, ax=axes[1])
axes[1].set_title('Renovation vs Price')

sns.barplot(x='season', y='price', data=df_fe, hue='season', palette='Set2', legend=False, ax=axes[2])
axes[2].set_title('Season vs Price')

for ax in axes:
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))

plt.tight_layout()
plt.show()


df_model = df_fe.copy()

df_model = df_model.drop(
    columns=['street', 'country', 'date', 'sqft_above', 'sqft_basement',
             'yr_built', 'yr_renovated', 'year', 'month', 'city', 'statezip'],
    errors='ignore'
)

df_model = pd.get_dummies(df_model, columns=['season'], drop_first=True)
df_model = df_model.apply(lambda col: col.astype(int) if col.dtype == 'bool' else col)

print('Shape after encoding:', df_model.shape)
print('Columns:', df_model.columns.tolist())

X = df_model.drop(columns=['price'])
y = df_model['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print('Train:', X_train.shape, 'Test:', X_test.shape)


X_train = X_train.copy()
X_test = X_test.copy()

zip_col = df_fe['statezip'].str.split(' ').str[1]
X_train['zip_raw'] = zip_col.reindex(X_train.index)
X_test['zip_raw'] = zip_col.reindex(X_test.index)
zip_mean = X_train.assign(price=y_train).groupby('zip_raw')['price'].mean()
X_train['zip_te'] = X_train['zip_raw'].map(zip_mean)
X_test['zip_te'] = X_test['zip_raw'].map(zip_mean).fillna(y_train.mean())
X_train = X_train.drop(columns=['zip_raw'], errors='ignore')
X_test = X_test.drop(columns=['zip_raw'], errors='ignore')

city_col = df_fe['city'].reindex(X_train.index)
city_col_test = df_fe['city'].reindex(X_test.index)
city_mean = X_train.assign(price=y_train, city=city_col).groupby('city')['price'].mean()
X_train['city_te'] = city_col.map(city_mean)
X_test['city_te'] = city_col_test.map(city_mean).fillna(y_train.mean())

print('After target encoding - Train:', X_train.shape)
print('Features:', X_train.columns.tolist())

train_df_corr = pd.concat([X_train, y_train], axis=1)
corr_price = train_df_corr.corr(numeric_only=True)['price'].abs().sort_values(ascending=False)
print('Feature correlations with price:')
print(corr_price)

selected_features = [f for f in corr_price.index if f != 'price' and corr_price[f] > 0.05]

must_keep = ['waterfront', 'view', 'is_renovated', 'house_age', 'has_basement', 'basement_ratio', 'zip_te', 'city_te']
selected_features = list(set(selected_features + must_keep))
selected_features = [f for f in selected_features if f in X_train.columns]

print(f'Selected {len(selected_features)} features')
print(selected_features)

corr_sel = train_df_corr[selected_features].corr().abs()
upper = corr_sel.where(np.triu(np.ones(corr_sel.shape), k=1).astype(bool))

to_drop = [col for col in upper.columns if any(upper[col] > 0.85)]

protect = ['sqft_living', 'zip_te', 'city_te', 'bathrooms']
to_drop = [f for f in to_drop if f not in protect]

final_features = [f for f in selected_features if f not in to_drop]

print(f'Dropped for multicollinearity: {to_drop}')
print(f'Final features: {len(final_features)}')
print(final_features)


final_corr = train_df_corr[final_features + ['price']].corr()
mask = np.triu(np.ones_like(final_corr, dtype=bool))

plt.figure(figsize=(14, 12))
sns.heatmap(final_corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0, square=True)
plt.title('Correlation Matrix - Final Features')
plt.show()


corr_price_final = train_df_corr[final_features + ['price']].corr()['price'].abs().drop('price').sort_values()

plt.figure(figsize=(8, max(6, len(final_features) * 0.35)))
corr_price_final.plot(kind='barh')
plt.title('Feature Correlation with Price')
plt.xlabel('Absolute Correlation')
plt.show()


X_train_sel = X_train[final_features]
X_test_sel = X_test[final_features]

scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_sel), columns=X_train_sel.columns, index=X_train_sel.index)
X_test_scaled = pd.DataFrame(scaler.transform(X_test_sel), columns=X_test_sel.columns, index=X_test_sel.index)

print('Scaled train:', X_train_scaled.shape)
print('Scaled test:', X_test_scaled.shape)


base_models = {
    'Ridge': Ridge(alpha=1.0),

    'Random Forest': RandomForestRegressor(
        n_estimators=500, max_depth=None, min_samples_leaf=2,
        max_features='sqrt', random_state=42
    ),

    'Extra Trees': ExtraTreesRegressor(
        n_estimators=500, max_depth=None, min_samples_leaf=2,
        max_features='sqrt', random_state=42
    ),

    'XGBoost': XGBRegressor(
        n_estimators=200, learning_rate=0.05, max_depth=5,
        subsample=0.8, colsample_bytree=0.8,
        objective='reg:squarederror', random_state=42
    ),

    'GBR': GradientBoostingRegressor(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        subsample=0.8, random_state=42
    ),

    'MLP': MLPRegressor(
        hidden_layer_sizes=(128, 64), max_iter=3000,
        alpha=0.001, learning_rate='adaptive',
        learning_rate_init=0.001, early_stopping=True,
        n_iter_no_change=30, random_state=42
    )
}


results = []
for name, model in base_models.items():
    print(f'Training {name}...')
    m = clone(model)
    m.fit(X_train_scaled, y_train)
    preds = m.predict(X_test_scaled)

    results.append({
        'Model': name,
        'R2': r2_score(y_test, preds),
        'RMSE': np.sqrt(mean_squared_error(y_test, preds)),
        'MAE': mean_absolute_error(y_test, preds)
    })

results_df = pd.DataFrame(results).sort_values('R2', ascending=False)
display(results_df)


kf = KFold(n_splits=5, shuffle=True, random_state=42)

train_oof = np.zeros((X_train_scaled.shape[0], len(base_models)))
test_preds = np.zeros((X_test_scaled.shape[0], len(base_models)))

for i, (name, model) in enumerate(base_models.items()):
    print(f'Generating OOF for {name}...')
    fold_test = np.zeros((X_test_scaled.shape[0], 5))

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train_scaled)):
        m = clone(model)
        m.fit(X_train_scaled.iloc[tr_idx], y_train.iloc[tr_idx])
        train_oof[val_idx, i] = m.predict(X_train_scaled.iloc[val_idx])
        fold_test[:, fold] = m.predict(X_test_scaled)

    test_preds[:, i] = fold_test.mean(axis=1)

X_meta_train = pd.DataFrame(train_oof, columns=base_models.keys(), index=X_train_scaled.index)
X_meta_test = pd.DataFrame(test_preds, columns=base_models.keys(), index=X_test_scaled.index)

print('OOF generation complete')


meta_v1 = Ridge(alpha=1.0)
meta_v1.fit(X_meta_train, y_train)
y_pred_v1 = meta_v1.predict(X_meta_test)

X_meta_train_full = pd.concat([X_meta_train, X_train_scaled], axis=1)
X_meta_test_full = pd.concat([X_meta_test, X_test_scaled], axis=1)

meta_v2 = Ridge(alpha=1.0)
meta_v2.fit(X_meta_train_full, y_train)
y_pred_v2 = meta_v2.predict(X_meta_test_full)

meta_v3 = LassoCV(cv=5, random_state=42)
meta_v3.fit(X_meta_train_full, y_train)
y_pred_v3 = meta_v3.predict(X_meta_test_full)

print('Stacking V1 (OOF only)')
print(f'  R2:   {r2_score(y_test, y_pred_v1):.4f}')
print(f'  RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_v1)):.0f}')

print('\nStacking V2 (OOF + passthrough, Ridge)')
print(f'  R2:   {r2_score(y_test, y_pred_v2):.4f}')
print(f'  RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_v2)):.0f}')

print('\nStacking V3 (OOF + passthrough, Lasso) ')
print(f'  R2:   {r2_score(y_test, y_pred_v3):.4f}')
print(f'  RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_v3)):.0f}')



print('V1 Meta-learner coefficients')
for name, coef in zip(base_models.keys(), meta_v1.coef_):
    print(f'  {name}: {coef:.4f}')

print('\nV3 Lasso non-zero coefficients')
v3_names = list(base_models.keys()) + list(X_train_scaled.columns)
for name, coef in zip(v3_names, meta_v3.coef_):
    if abs(coef) > 0.001:
        print(f'  {name}: {coef:.4f}')


all_results = results.copy()
all_results.append({'Model': 'Stacking V1', 'R2': r2_score(y_test, y_pred_v1), 'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_v1)), 'MAE': mean_absolute_error(y_test, y_pred_v1)})
all_results.append({'Model': 'Stacking V2', 'R2': r2_score(y_test, y_pred_v2), 'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_v2)), 'MAE': mean_absolute_error(y_test, y_pred_v2)})
all_results.append({'Model': 'Stacking V3', 'R2': r2_score(y_test, y_pred_v3), 'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_v3)), 'MAE': mean_absolute_error(y_test, y_pred_v3)})

all_df = pd.DataFrame(all_results).sort_values('R2', ascending=False)
display(all_df)

plt.figure(figsize=(10, 6))
colors = ['#e74c3c' if 'Stacking' in m else '#3498db' for m in all_df['Model']]
plt.barh(all_df['Model'], all_df['R2'], color=colors)
plt.xlabel('R² Score')
plt.title('Model Comparison: Individual Models vs Stacking Ensembles')
plt.xlim(all_df['R2'].min() - 0.02, all_df['R2'].max() + 0.01)
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()



best_v = np.argmax([r2_score(y_test, y_pred_v1), r2_score(y_test, y_pred_v2), r2_score(y_test, y_pred_v3)])
best_stack_preds = [y_pred_v1, y_pred_v2, y_pred_v3][best_v]
best_stack_r2 = r2_score(y_test, best_stack_preds)

ablation_results = []
model_names = list(base_models.keys())

for remove in model_names:
    print(f'Ablation: removing {remove}...')
    keep_idx = [j for j, n in enumerate(model_names) if n != remove]

    X_ab_train = X_meta_train.iloc[:, keep_idx]
    X_ab_test = X_meta_test.iloc[:, keep_idx]

    ab_meta = Ridge(alpha=1.0)
    ab_meta.fit(X_ab_train, y_train)
    ab_preds = ab_meta.predict(X_ab_test)

    ablation_results.append({
        'Removed': remove,
        'R2': r2_score(y_test, ab_preds),
        'RMSE': np.sqrt(mean_squared_error(y_test, ab_preds)),
        'R2_drop': best_stack_r2 - r2_score(y_test, ab_preds)
    })

ablation_results.append({'Removed': 'None (Full)', 'R2': best_stack_r2, 'RMSE': np.sqrt(mean_squared_error(y_test, best_stack_preds)), 'R2_drop': 0})

ab_df = pd.DataFrame(ablation_results).sort_values('R2_drop', ascending=False)
display(ab_df)

plt.figure(figsize=(8, 5))
ab_plot = ab_df[ab_df['Removed'] != 'None (Full)'].sort_values('R2_drop')
colors = ['#e74c3c' if d > 0 else '#2ecc71' for d in ab_plot['R2_drop']]
plt.barh(ab_plot['Removed'], ab_plot['R2_drop'], color=colors)
plt.xlabel('R² Drop (higher = more important)')
plt.title('Ablation Study: Contribution of Each Base Model')
plt.axvline(x=0, color='black', linewidth=0.5)
plt.tight_layout()
plt.show()

rf_imp = RandomForestRegressor(n_estimators=200, random_state=42)
rf_imp.fit(X_train_scaled, y_train)

importances = pd.Series(rf_imp.feature_importances_, index=X_train_scaled.columns)

plt.figure(figsize=(8, max(6, len(final_features) * 0.35)))
importances.sort_values().plot(kind='barh')
plt.title('Random Forest Feature Importance')
plt.xlabel('Importance')
plt.tight_layout()
plt.show()

df_raw = df_original.copy()
df_raw = df_raw.drop(columns=['street', 'city', 'date', 'country', 'statezip'])

X_raw = df_raw.drop(columns=['price'])
y_raw = df_raw['price']

X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(X_raw, y_raw, test_size=0.2, random_state=42)

compare_results = []

for name, model in [('Random Forest', RandomForestRegressor(n_estimators=200, max_depth=None, random_state=42)),
                     ('XGBoost', XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, objective='reg:squarederror', random_state=42))]:
    m = clone(model)
    m.fit(X_train_raw, y_train_raw)
    p = m.predict(X_test_raw)
    compare_results.append({'Stage': 'Before', 'Model': name, 'R2': r2_score(y_test_raw, p), 'RMSE': np.sqrt(mean_squared_error(y_test_raw, p))})

    m2 = clone(model)
    m2.fit(X_train_scaled, y_train)
    p2 = m2.predict(X_test_scaled)
    compare_results.append({'Stage': 'After', 'Model': name, 'R2': r2_score(y_test, p2), 'RMSE': np.sqrt(mean_squared_error(y_test, p2))})

compare_df = pd.DataFrame(compare_results)
display(compare_df)

plt.figure(figsize=(8, 5))
sns.barplot(data=compare_df, x='Model', y='R2', hue='Stage')
plt.title('Impact of Preprocessing on Model Performance')
plt.ylabel('R² Score')
plt.show()



best_preds = [y_pred_v1, y_pred_v2, y_pred_v3][best_v]
best_name = ['V1', 'V2', 'V3'][best_v]
residuals = y_test - best_preds

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].scatter(best_preds, y_test.values, alpha=0.3, s=10)
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')
axes[0].set_title(f'Predicted vs Actual (Stacking {best_name})')

axes[1].scatter(best_preds, residuals, alpha=0.3, s=10)
axes[1].axhline(y=0, color='r', linestyle='--')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Residual')
axes[1].set_title('Residual Plot')

sns.histplot(residuals, bins=50, kde=True, ax=axes[2])
axes[2].set_title(f'Residual Distribution (mean={residuals.mean():.0f})')

plt.tight_layout()
plt.show()




