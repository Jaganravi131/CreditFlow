import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

print("Loading processed data...")
try:
    data = pd.read_csv('processed_attribution_data.csv')
    print(f"Processed data loaded successfully! Shape: {data.shape}")
except FileNotFoundError:
    print("Processed data not found. Please run feature_engineering.py first.")
    exit(1)

# Define categorical and numeric features
categorical_features = ['campaign']
numeric_features = [col for col in data.columns if col not in categorical_features + ['conversion', 'attribution']]

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Split data for conversion modeling
print("Splitting data for conversion modeling...")
X_conv = data.drop(['conversion', 'attribution'], axis=1)
y_conv = data['conversion']
X_train_conv, X_test_conv, y_train_conv, y_test_conv = train_test_split(
    X_conv, y_conv, test_size=0.2, random_state=42
)

# Split data for attribution modeling (only using converted impressions)
converted_data = data[data['conversion'] == 1]
if len(converted_data) > 0:
    print(f"Splitting data for attribution modeling... ({len(converted_data)} conversions)")
    X_attr = converted_data.drop(['conversion', 'attribution'], axis=1)
    y_attr = converted_data['attribution']
    X_train_attr, X_test_attr, y_train_attr, y_test_attr = train_test_split(
        X_attr, y_attr, test_size=0.2, random_state=42
    )

# 1. Conversion Prediction Model
print("Training conversion prediction model...")
conversion_model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', GradientBoostingClassifier(n_estimators=100, random_state=42))
])

conversion_model.fit(X_train_conv, y_train_conv)

# Evaluate conversion model
y_pred_conv_proba = conversion_model.predict_proba(X_test_conv)[:, 1]
conv_auc = roc_auc_score(y_test_conv, y_pred_conv_proba)
print(f"Conversion model AUC: {conv_auc:.4f}")

# Plot conversion probability distribution
plt.figure(figsize=(10, 6))
plt.hist(y_pred_conv_proba, bins=50)
plt.title('Distribution of Conversion Probabilities')
plt.xlabel('Conversion Probability')
plt.ylabel('Count')
plt.savefig('conversion_probability_distribution.png')
print("Created visualization: conversion_probability_distribution.png")

# 2. Attribution Model (for converted impressions)
if len(converted_data) > 0:
    print("Training attribution prediction model...")
    attribution_model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', GradientBoostingClassifier(n_estimators=100, random_state=42))
    ])
    
    attribution_model.fit(X_train_attr, y_train_attr)
    
    # Evaluate attribution model
    y_pred_attr_proba = attribution_model.predict_proba(X_test_attr)[:, 1]
    attr_auc = roc_auc_score(y_test_attr, y_pred_attr_proba)
    print(f"Attribution model AUC: {attr_auc:.4f}")
    
    # Plot attribution probability distribution
    plt.figure(figsize=(10, 6))
    plt.hist(y_pred_attr_proba, bins=50)
    plt.title('Distribution of Attribution Probabilities')
    plt.xlabel('Attribution Probability')
    plt.ylabel('Count')
    plt.savefig('attribution_probability_distribution.png')
    print("Created visualization: attribution_probability_distribution.png")
else:
    print("No conversions found in the dataset, skipping attribution model.")

# 3. Multi-touch Attribution Model
print("Creating multi-touch attribution model...")

# Function to calculate attribution weights based on position and time
def calculate_attribution_weights(data):
    # Get only clicked impressions that led to conversions
    clicked_conversions = data[(data['conversion'] == 1) & (data['click'] == 1)].copy()
    
    if len(clicked_conversions) == 0:
        print("No clicked conversions found, using default weights.")
        return {'first': 0.4, 'last': 0.4, 'linear': 0.2}
    
    # Group by conversion_id to get all clicks in a conversion path
    conversion_paths = clicked_conversions.groupby('conversion_id')
    
    # Calculate position-based weights
    first_click_weight = 0.4
    last_click_weight = 0.4
    linear_weight = 0.2  # Distributed among middle clicks
    
    # Apply weights to each conversion path
    attribution_results = []
    
    for conv_id, path in conversion_paths:
        # Sort by timestamp to get chronological order
        path = path.sort_values('timestamp')
        
        # Number of clicks in this path
        n_clicks = len(path)
        
        if n_clicks == 1:
            # If only one click, it gets full credit
            path['attribution_weight'] = 1.0
        else:
            # First click
            path.iloc[0, path.columns.get_loc('attribution_weight')] = first_click_weight
            
            # Last click
            path.iloc[-1, path.columns.get_loc('attribution_weight')] = last_click_weight
            
            # Middle clicks (if any)
            if n_clicks > 2:
                middle_weight = linear_weight / (n_clicks - 2)
                for i in range(1, n_clicks - 1):
                    path.iloc[i, path.columns.get_loc('attribution_weight')] = middle_weight
        
        attribution_results.append(path)
    
    # Combine all results
    if attribution_results:
        all_attributions = pd.concat(attribution_results)
        return all_attributions
    else:
        print("No multi-click conversion paths found, using default weights.")
        return {'first': 0.4, 'last': 0.4, 'linear': 0.2}

# Try to calculate attribution weights if we have conversion_id in the data
if 'conversion_id' in data.columns:
    try:
        data['attribution_weight'] = 0.0  # Initialize column
        attribution_weights = calculate_attribution_weights(data)
        print("Multi-touch attribution weights calculated.")
    except Exception as e:
        print(f"Error calculating multi-touch attribution: {e}")
        print("Using default attribution model.")
else:
    print("conversion_id not found in data, skipping multi-touch attribution.")

# Save models
print("Saving models...")
os.makedirs('models', exist_ok=True)

with open('models/conversion_model.pkl', 'wb') as f:
    pickle.dump(conversion_model, f)
print("Saved conversion model to models/conversion_model.pkl")

if len(converted_data) > 0:
    with open('models/attribution_model.pkl', 'wb') as f:
        pickle.dump(attribution_model, f)
    print("Saved attribution model to models/attribution_model.pkl")

# Save feature names for later use
with open('models/feature_names.pkl', 'wb') as f:
    pickle.dump({
        'categorical_features': categorical_features,
        'numeric_features': numeric_features
    }, f)
print("Saved feature names to models/feature_names.pkl")

print("Attribution modeling completed successfully!")
