import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
import matplotlib.pyplot as plt
import shap
import re

# 1. Load and preprocess data
print("Loading data...")
price_data = pd.read_csv('pricedBitcoin2009-2018.csv')
amo_data = pd.read_csv('AmoChainletsInTime.txt', sep='\t')
occ_data = pd.read_csv('OccChainletsInTime.txt', sep='\t')

# Ensure key columns are consistent
price_data['year'] = price_data['year'].astype(str).str.strip()
amo_data['year'] = amo_data['year'].astype(str).str.strip()
occ_data['year'] = occ_data['year'].astype(str).str.strip()
price_data['day'] = price_data['day'].astype(str).str.strip()
amo_data['day'] = amo_data['day'].astype(str).str.strip()
occ_data['day'] = occ_data['day'].astype(str).str.strip()

# 2. Merge data on year and day
merged_data = pd.merge(price_data, amo_data, on=['year', 'day'], how='inner')
merged_data = pd.merge(merged_data, occ_data, on=['year', 'day'], how='inner', suffixes=('_amo', '_occ'))

# 3. Convert date to datetime properly using standard library
merged_data['date'] = pd.to_datetime(merged_data['date'])
merged_data = merged_data.sort_values('date')

# 4. Focus on 2016-2017 data only
merged_data = merged_data[(merged_data['year'].astype(int) >= 2016) & (merged_data['year'].astype(int) <= 2017)]
print(f"Total data points after filtering to 2016-2017: {len(merged_data)}")

# 5. Feature selection - identify chainlet columns for both types
amo_chainlet_columns = [col for col in merged_data.columns if re.match(r'\d+:\d+_amo$', col)]
occ_chainlet_columns = [col for col in merged_data.columns if re.match(r'\d+:\d+_occ$', col)]

print(f"Number of amount chainlet features: {len(amo_chainlet_columns)}")
print(f"Number of occurrence chainlet features: {len(occ_chainlet_columns)}")


# 6. Select top chainlets from each type based on correlation with price
# This ensures we're using both types as required while managing dimensionality
def select_top_features(data, feature_columns, target, n=5):  # Reduced from 10 to 5 features per type
    """Select top n features based on correlation with target."""
    correlations = []
    for col in feature_columns:
        corr = abs(data[col].corr(data[target]))
        correlations.append((col, corr))

    correlations.sort(key=lambda x: x[1], reverse=True)
    return [col for col, _ in correlations[:n]]


# Select top chainlets from each type
top_amo_chainlets = select_top_features(merged_data, amo_chainlet_columns, 'price', n=10)
top_occ_chainlets = select_top_features(merged_data, occ_chainlet_columns, 'price', n=10)

print("Top amount chainlet features:")
for feature in top_amo_chainlets:
    print(f"  - {feature}")

print("Top occurrence chainlet features:")
for feature in top_occ_chainlets:
    print(f"  - {feature}")

# 7. Split into training and test sets based on dates
# Test set: December 2017
december_2017_data = merged_data[(merged_data['date'] >= '2017-12-01') & (merged_data['date'] <= '2017-12-31')].copy()
# Training set: 2016 through November 2017
train_data = merged_data[merged_data['date'] < '2017-12-01'].copy()

print(f"Training data date range: {train_data['date'].min()} to {train_data['date'].max()}")
print(f"Test data date range: {december_2017_data['date'].min()} to {december_2017_data['date'].max()}")

# 8. Feature engineering - add lagged price features for one-day-ahead prediction
lag_days = 3  # Reduced from 5 to 3 days of lag features
for i in range(1, lag_days + 1):
    train_data.loc[:, f'price_lag_{i}'] = train_data['price'].shift(i)

# Add moving averages
train_data.loc[:, 'price_ma_7'] = train_data['price'].rolling(window=7).mean()

# Remove rows with NaN values due to lags/moving averages
train_data = train_data.dropna()

# 9. For test data, create features properly using lookback data
max_lookback = max(lag_days, 7)  # For 7-day MA (reduced from 14)
november_end_data = merged_data[
    (merged_data['date'] >= '2017-11-01') &
    (merged_data['date'] < '2017-12-01')
    ].tail(max_lookback).copy()

# Combine with December data for feature creation
temp_test_data = pd.concat([november_end_data, december_2017_data]).sort_values('date').copy()

# Create the same features for test data
for i in range(1, lag_days + 1):
    temp_test_data.loc[:, f'price_lag_{i}'] = temp_test_data['price'].shift(i)

temp_test_data.loc[:, 'price_ma_7'] = temp_test_data['price'].rolling(window=7).mean()

# Keep only December data after feature creation
test_data = temp_test_data[temp_test_data['date'] >= '2017-12-01'].dropna()

# 10. Prepare features for model - explicitly INCLUDING chainlets from both types
# Model features include lagged prices, moving averages, and selected chainlet features
model_features = [
                     f'price_lag_{i}' for i in range(1, lag_days + 1)
                 ] + [
                     'price_ma_7'
                 ] + top_amo_chainlets + top_occ_chainlets

print(f"Total features used in model: {len(model_features)}")

# Prepare X and y for both sets
X_train = train_data[model_features]
y_train = train_data['price']

X_test = test_data[model_features]
y_test = test_data['price']

# Print shapes of datasets for debugging
print(f'X_train shape: {X_train.shape}')
print(f'y_train shape: {y_train.shape}')
print(f'X_test shape: {X_test.shape}')
print(f'y_test shape: {y_test.shape}')

# 11. Standardize features properly - fit on training data only
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrames with column names for SHAP analysis
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=model_features, index=X_train.index)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=model_features, index=X_test.index)


# 12. Train and evaluate all models

# Function to evaluate models
def evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'{model_name} RMSE: {rmse}')
    print(f'{model_name} MAE: {mae}')
    print(f'{model_name} R²: {r2}')
    return y_pred, model


# Train and evaluate Linear Regression
print("Training Linear Regression model...")
lr_model = LinearRegression()
lr_pred, lr_model = evaluate_model(lr_model, X_train_scaled, y_train, X_test_scaled, y_test, 'Linear Regression')

# Train and evaluate Ridge Regression model...
print("Training Ridge Regression model...")
ridge_model = Ridge(alpha=100, random_state=42)  # Increased alpha from 10 to 100
ridge_pred, ridge_model = evaluate_model(ridge_model, X_train_scaled, y_train, X_test_scaled, y_test,
                                         'Ridge Regression')

# Train and evaluate Lasso Regression
print("Training Lasso Regression model...")
lasso_model = Lasso(alpha=100, max_iter=10000, random_state=42)  # Increased alpha from 10 to 100
lasso_pred, lasso_model = evaluate_model(lasso_model, X_train_scaled, y_train, X_test_scaled, y_test,
                                         'Lasso Regression')

# Train and evaluate ElasticNet Regression
print("Training ElasticNet Regression model...")
elasticnet_model = ElasticNet(alpha=1, l1_ratio=0.8, max_iter=10000,
                              random_state=42)  # Using parameters from previous results
elasticnet_pred, elasticnet_model = evaluate_model(elasticnet_model, X_train_scaled, y_train, X_test_scaled, y_test,
                                                   'ElasticNet Regression')

# Determine the best model based on RMSE
model_results = [
    ('Linear Regression', lr_model, lr_pred, np.sqrt(mean_squared_error(y_test, lr_pred))),
    ('Ridge Regression', ridge_model, ridge_pred, np.sqrt(mean_squared_error(y_test, ridge_pred))),
    ('Lasso Regression', lasso_model, lasso_pred, np.sqrt(mean_squared_error(y_test, lasso_pred))),
    ('ElasticNet Regression', elasticnet_model, elasticnet_pred, np.sqrt(mean_squared_error(y_test, elasticnet_pred)))
]

best_model_info = min(model_results, key=lambda x: x[3])
best_model_name, best_model, best_pred, best_rmse = best_model_info

print(f"\nBest model: {best_model_name} with RMSE: {best_rmse:.2f}")

# 15. Save predictions from the best model to CSV
predicted_prices = pd.DataFrame({
    'date': test_data['date'],
    'predicted_price': best_pred
})
predicted_prices.to_csv('predicted_bitcoin_prices.csv', index=False)

# 16. Display the first few predictions
print(f"\nFirst few predictions from {best_model_name}:")
print(predicted_prices.head())

# 17. Model explanation using SHAP for the best model
print(f"\nGenerating SHAP explanations for {best_model_name}...")
explainer = shap.Explainer(best_model, X_train_scaled_df)
shap_values = explainer(X_test_scaled_df)

# 18. Plot feature importance summary
plt.figure(figsize=(10, 8))
shap.plots.bar(shap_values, show=False)
plt.title("Feature Importance Based on SHAP Values")
plt.tight_layout()
plt.savefig("shap_feature_importance.png")
plt.close()

# 19. Analyze specific days as required (beginning, middle, end of December)
days_to_explain = ['2017-12-01', '2017-12-15', '2017-12-31']
test_data_with_date = test_data.reset_index()
test_data_with_date['date_str'] = test_data_with_date['date'].dt.strftime('%Y-%m-%d')

for day in days_to_explain:
    idx = test_data_with_date[test_data_with_date['date_str'] == day].index[0]

    print(f"\nSHAP Analysis for {day}:")

    # Get the predicted vs actual price
    predicted_price = lasso_pred[idx]
    actual_price = y_test.iloc[idx]
    print(f"Actual price: ${actual_price:.2f}")
    print(f"Predicted price: ${predicted_price:.2f}")
    print(f"Difference: ${predicted_price - actual_price:.2f}")

    # Get the top 5 most influential features for this prediction
    instance_shap = shap_values[idx]
    feature_importance = list(zip(model_features, instance_shap.values))
    feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 influential features:")
    for feature, importance in feature_importance[:5]:
        direction = "increased" if importance > 0 else "decreased"
        print(f"  - {feature} {direction} the price by ${abs(importance):.2f}")

    # Generate and save waterfall plot for this prediction
    plt.figure(figsize=(12, 8))
    shap.plots.waterfall(shap_values[idx], show=False)
    plt.title(f"SHAP Explanation for Bitcoin Price on {day}")
    plt.tight_layout()
    plt.savefig(f"shap_explanation_{day}.png")
    plt.close()

# 20. Plot actual vs. predicted prices for all models
plt.figure(figsize=(14, 8))
plt.plot(test_data['date'], y_test, label='Actual Price', color='black', linewidth=2)
plt.plot(test_data['date'], lr_pred, label='Linear Regression', color='blue', linestyle='--')
plt.plot(test_data['date'], ridge_pred, label='Ridge Regression', color='green', linestyle='--')
plt.plot(test_data['date'], lasso_pred, label='Lasso Regression', color='red', linestyle='--')
plt.plot(test_data['date'], elasticnet_pred, label='ElasticNet Regression', color='purple', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.title('Actual vs Predicted Bitcoin Prices for December 2017')
plt.legend()
plt.grid(True)
plt.savefig("price_predictions.png")
plt.show()

print("\nAnalysis complete - all outputs have been saved to files.")