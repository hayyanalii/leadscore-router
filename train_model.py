"""
train_model.py
Loads call_data.csv, trains a Random Forest model to predict conversion,
and saves it as 'model.pkl' for the API to use.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

print("Loading data...")
df = pd.read_csv("call_data.csv")
print(f"✅ Loaded {len(df)} rows")

# --- Step 1: Separate features (X) from target (y) ---
# X = everything the model uses to make a decision
# y = what we want to predict

X = df.drop("converted", axis=1)   # all columns EXCEPT 'converted'
y = df["converted"]                 # the target we want to predict

print(f"Features: {list(X.columns)}")
print(f"Target: converted (0 or 1)")

# --- Step 2: Split data: 80% train, 20% test ---
# We train on 80% and see how it performs on the 20% it has never seen.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training set: {len(X_train)} rows")
print(f"Test set:     {len(X_test)} rows")

# --- Step 3: Train the model ---
print("\nTraining Random Forest model...")
model = RandomForestClassifier(
    n_estimators=100,     # 100 decision trees voting together
    max_depth=10,         # limit depth so it doesn't memorize
    random_state=42,
    n_jobs=-1             # use all CPU cores
)
model.fit(X_train, y_train)
print("✅ Model trained!")

# --- Step 4: Evaluate on the test set ---
print("\nEvaluating on test set...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Accuracy: {accuracy * 100:.2f}%")
print("\nDetailed report:")
print(classification_report(y_test, y_pred, target_names=["Not Converted", "Converted"]))

# --- Step 5: Show which features matter most ---
print("Feature importance (what drives conversion):")
importances = model.feature_importances_
for feat, imp in sorted(zip(X.columns, importances), key=lambda x: -x[1]):
    print(f"  {feat:25s} {imp * 100:.1f}%")

# --- Step 6: Save the model to disk ---
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
print("\n✅ Model saved as 'model.pkl'")

# --- Step 7: Test a single prediction (sanity check) ---
sample = X_test.iloc[0:1]
prediction = model.predict(sample)[0]
probability = model.predict_proba(sample)[0][1]

print("\n--- Sample Prediction ---")
print(f"Input: {sample.to_dict(orient='records')[0]}")
print(f"Prediction: {'CONVERT' if prediction == 1 else 'NO CONVERT'}")
print(f"Conversion probability: {probability * 100:.1f}%")