import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from evaluate import evaluate
import sys

# ========================== Load and Prepare Data ==========================
df = pd.read_csv('./cleaned_data.csv')
df = df.dropna(subset=['Income'])

X = df.drop(['ID', 'Income'], axis=1)
y = df['Income']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Redirect print output to a file
sys.stdout = open('./random_forest_output.txt', 'w')

# ========================== Random Forest Regression with GridSearch ==========================
param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [None, 5, 10]
}

rf = RandomForestRegressor(random_state=42)
grid_search_rf = GridSearchCV(rf, param_grid_rf, cv=5, scoring='r2', n_jobs=-1)
grid_search_rf.fit(X_train, y_train)

# Best Random Forest model
best_rf = grid_search_rf.best_estimator_

# Predict on test set
y_pred_rf = best_rf.predict(X_test)

# Evaluate
evaluate(y_test, y_pred_rf, "Random Forest Regression (Tuned)")

# Cross-validation score
cv_scores_rf = cross_val_score(best_rf, X_train, y_train, cv=5, scoring='r2')

print("--- Best Random Forest Parameters ---")
print(grid_search_rf.best_params_)
print(f"Random Forest CV Mean R2: {np.mean(cv_scores_rf):.4f}")

# ========================== Feature Importance Plot ==========================
importances = best_rf.feature_importances_
feature_names = df.drop(['ID', 'Income'], axis=1).columns
indices = np.argsort(importances)

plt.figure(figsize=(10, 6))
plt.title('Random Forest Feature Importances')
bars = plt.barh(range(len(indices)), importances[indices], align='center')
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel('Relative Importance')

# ===== Add value labels =====
for bar in bars:
    plt.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
             f"{bar.get_width():.4f}",
             va='center', fontsize=8)

plt.savefig('./feature_importance_rf.png', bbox_inches='tight', dpi=300)
plt.show()

# Close the text file
sys.stdout.close()
