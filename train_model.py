import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import joblib

# --- 1. Load the Dataset ---
df = pd.read_csv('1970-2021_DISASTERS.xlsx - emdat data.csv')

# --- 2. Data Filtering and Preprocessing ---
disasters_to_keep = ['Flood', 'Storm', 'Landslide', 'Wildfire']
df_filtered = df[df['Disaster Type'].isin(disasters_to_keep)].copy()
print(f"Records kept after filtering for specified disasters: {len(df_filtered)}")

# **FIX: Removed Disaster Group and Subgroup from columns to use**
columns_to_use = [
    'Year', 'Country', 'Region', 'Continent', 'Start Month', 'Start Day', 'Disaster Type'
]
df_filtered = df_filtered[columns_to_use]
df_filtered.dropna(inplace=True)
df_filtered['Start Month'] = pd.to_numeric(df_filtered['Start Month'], errors='coerce').astype(int)
df_filtered['Start Day'] = pd.to_numeric(df_filtered['Start Day'], errors='coerce').astype(int)
print(f"Final number of records after cleaning: {len(df_filtered)}")

# --- 3. Feature Engineering ---
df_filtered['month_sin'] = np.sin(2 * np.pi * df_filtered['Start Month']/12)
df_filtered['month_cos'] = np.cos(2 * np.pi * df_filtered['Start Month']/12)
df_filtered['day_sin'] = np.sin(2 * np.pi * df_filtered['Start Day']/31)
df_filtered['day_cos'] = np.cos(2 * np.pi * df_filtered['Start Day']/31)

country_encoder = LabelEncoder()
region_encoder = LabelEncoder()
continent_encoder = LabelEncoder()
disaster_encoder = LabelEncoder()

df_filtered['Country_Encoded'] = country_encoder.fit_transform(df_filtered['Country'])
df_filtered['Region_Encoded'] = region_encoder.fit_transform(df_filtered['Region'])
df_filtered['Continent_Encoded'] = continent_encoder.fit_transform(df_filtered['Continent'])
df_filtered['Disaster_Type_Encoded'] = disaster_encoder.fit_transform(df_filtered['Disaster Type'])

# --- 4. Model Training with Hyperparameter Tuning ---
# **FIX: Removed leaky features from the features list**
features = [
    'Year', 'Country_Encoded', 'Region_Encoded', 'Continent_Encoded',
    'month_sin', 'month_cos', 'day_sin', 'day_cos'
]
X = df_filtered[features]
y = df_filtered['Disaster_Type_Encoded']

num_classes = len(disaster_encoder.classes_)
print(f"Total number of disaster types for training: {num_classes} ({disaster_encoder.classes_})")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 8],
    'learning_rate': [0.05, 0.1],
    'colsample_bytree': [0.7, 1.0]
}

xgb_model = XGBClassifier(
    objective='multi:softprob',
    num_class=num_classes,
    eval_metric='mlogloss',
    use_label_encoder=False
)

grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)

print("\nStarting hyperparameter tuning on predictive features...")
grid_search.fit(X_train, y_train)

print(f"\nBest parameters found: {grid_search.best_params_}")
model = grid_search.best_estimator_
print("\nModel training with best parameters completed successfully!")

# --- 5. Evaluate Model Accuracy ---
print("\nEvaluating final model performance...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"🚀 Final (Honest) Model Accuracy: {accuracy * 100:.2f}%")

# --- 6. Save the Final Model and Encoders ---
model.save_model("disaster_model.json")
joblib.dump(country_encoder, 'country_encoder.joblib')
joblib.dump(region_encoder, 'region_encoder.joblib')
joblib.dump(continent_encoder, 'continent_encoder.joblib')
joblib.dump(disaster_encoder, 'disaster_encoder.joblib')

print("\nTruly predictive model and encoders have been saved.")


