import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from evaluate import evaluate
import sys
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('./cleaned_data.csv')
df = df.dropna(subset=['Income'])

# Separate features and target
X = df.drop(['ID', 'Income'], axis=1)
y = df['Income']

# Train/Test split FIRST
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

sys.stdout = open('./linear_regression_output.txt', 'w')

# ========================== Linear Regression (baseline) ==========================
lr = LinearRegression()

cv_scores_lr = cross_val_score(lr, X_train, y_train, cv=5, scoring='r2')
print("--- Cross-Validation R² Scores (Linear) ---")
print(f"Linear Regression CV Mean R2: {np.mean(cv_scores_lr):.4f}")

lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

evaluate(y_test, y_pred_lr, "Linear Regression")

# ========================== Ridge Regression with GridSearch ==========================
ridge = Ridge()

param_grid_ridge = {'alpha': [0.01, 0.1, 1, 10, 100]}
grid_ridge = GridSearchCV(ridge, param_grid_ridge, cv=5, scoring='r2')
grid_ridge.fit(X_train, y_train)

best_ridge = grid_ridge.best_estimator_
y_pred_ridge = best_ridge.predict(X_test)

evaluate(y_test, y_pred_ridge, f"Tuned Ridge Regression (alpha={grid_ridge.best_params_['alpha']})")

print("--- Best Ridge Parameters ---")
print(grid_ridge.best_params_)
print(f"Tuned Ridge Regression CV Mean R2: {grid_ridge.best_score_:.4f}")

# ========================== Lasso Regression with GridSearch ==========================
lasso = Lasso(max_iter=10000)

param_grid_lasso = {'alpha': [0.001, 0.01, 0.1, 1, 10]}
grid_lasso = GridSearchCV(lasso, param_grid_lasso, cv=5, scoring='r2')
grid_lasso.fit(X_train, y_train)

best_lasso = grid_lasso.best_estimator_
y_pred_lasso = best_lasso.predict(X_test)

evaluate(y_test, y_pred_lasso, f"Tuned Lasso Regression (alpha={grid_lasso.best_params_['alpha']})")

print("--- Best Lasso Parameters ---")
print(grid_lasso.best_params_)
print(f"Tuned Lasso Regression CV Mean R2: {grid_lasso.best_score_:.4f}")

# ========================== Model Comparison Table ==========================
results = pd.DataFrame({
    'Model': ['Linear', 'Tuned Ridge', 'Tuned Lasso'],
    'CV Mean R²': [np.mean(cv_scores_lr), grid_ridge.best_score_, grid_lasso.best_score_]
})

print("--- Model Comparison ---")
print(results)

sys.stdout.close()

models = ['Linear', 'Tuned Ridge', 'Tuned Lasso']
scores = [np.mean(cv_scores_lr), grid_ridge.best_score_, grid_lasso.best_score_]

plt.figure(figsize=(8, 5))
bars = plt.bar(models, scores)
plt.ylim(0, 1)  # Set y-axis limit between 0 and 1 since R² is between 0 and 1
plt.ylabel('Cross-Validation Mean R² Score')
plt.title('Model Comparison: Linear vs Tuned Ridge vs Tuned Lasso')
plt.grid(axis='y')

# ===== Add value labels on top of bars =====
for bar, score in zip(bars, scores):
    plt.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.01,  # 0.01 offset above bar
             f"{score:.4f}",
             ha='center', va='bottom', fontsize=10)

# Save the plot
plt.savefig('./lin_reg_comparison_bar_chart.png', bbox_inches='tight', dpi=300)
plt.show()