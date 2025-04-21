import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from evaluate import evaluate
import matplotlib.pyplot as plt
import sys
import warnings
from sklearn.exceptions import ConvergenceWarning

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
sys.stdout = open('./poly_regression_output.txt', 'w')

# ========================== Polynomial Regression (baseline) ==========================
degrees = [2, 3, 4]
cv_scores_linear = []

for d in degrees:
    model_linear = make_pipeline(PolynomialFeatures(degree=d), LinearRegression())
    scores_linear = cross_val_score(model_linear, X_train, y_train, cv=5, scoring='r2')
    cv_scores_linear.append(np.mean(scores_linear))

best_degree_linear = degrees[np.argmax(cv_scores_linear)]

# ========================== Polynomial Regression + Ridge Regression with GridSearch ==========================
ridge_pipeline = make_pipeline(PolynomialFeatures(), Ridge())

param_grid_ridge = {
    'polynomialfeatures__degree': [2, 3, 4],
    'ridge__alpha': [0.01, 0.1, 1, 10, 100]
}

grid_ridge = GridSearchCV(ridge_pipeline, param_grid_ridge, cv=5, scoring='r2', n_jobs=-1)
grid_ridge.fit(X_train, y_train)

best_ridge_model = grid_ridge.best_estimator_

# ========================== Polynomial Regression + Lasso Regression with Manual GridSearch and Warning Capture ==========================
print("\n--- Checking Lasso Convergence ---")

lasso_results = []
best_score = -np.inf
best_lasso_model = None
best_params_lasso = {}

for degree in [2, 3, 4]:
    for alpha in [0.001, 0.01, 0.1, 1, 10]:
        model = make_pipeline(
            PolynomialFeatures(degree=degree),
            Lasso(alpha=alpha, max_iter=50000)
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always', ConvergenceWarning)
            scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
            mean_score = np.mean(scores)
            
            converged = not any(issubclass(warning.category, ConvergenceWarning) for warning in w)
            if not converged:
                print(f"ConvergenceWarning: Degree={degree}, Alpha={alpha}")
            
            # Save result
            lasso_results.append({
                'degree': degree,
                'alpha': alpha,
                'cv_mean_r2': mean_score,
                'converged': converged
            })
            
            # Track best model
            if mean_score > best_score:
                best_score = mean_score
                best_lasso_model = model
                best_params_lasso = {'degree': degree, 'alpha': alpha}

# Fit the best Lasso model
best_lasso_model.fit(X_train, y_train)

# ========================== Print CV Summary ==========================
print("--- Polynomial Degree Comparison (Linear) ---")
for i, d in enumerate(degrees):
    print(f"Degree {d}: LinearR CV Mean R2 = {cv_scores_linear[i]:.4f}")

print(f"\nBest Degree (LinearRegression): {best_degree_linear}")
print(f"Best Ridge Parameters: {grid_ridge.best_params_}")
print(f"Best Ridge CV Mean R2: {grid_ridge.best_score_:.4f}")
print(f"Best Lasso Parameters: {best_params_lasso}")
print(f"Best Lasso CV Mean R2: {best_score:.4f}")

# ========================== Retrain Best Models and Evaluate ==========================
# Best Linear Model
best_linear = make_pipeline(PolynomialFeatures(degree=best_degree_linear), LinearRegression())
best_linear.fit(X_train, y_train)
y_pred_best_linear = best_linear.predict(X_test)

# Best Ridge Model
y_pred_best_ridge = best_ridge_model.predict(X_test)

# Best Lasso Model
y_pred_best_lasso = best_lasso_model.predict(X_test)

# Evaluate all models
evaluate(y_test, y_pred_best_linear, f"Best Polynomial Linear Regression (Degree {best_degree_linear})")
evaluate(y_test, y_pred_best_ridge, f"Tuned Ridge Regression (Degree {grid_ridge.best_params_['polynomialfeatures__degree']}, Alpha {grid_ridge.best_params_['ridge__alpha']})")
evaluate(y_test, y_pred_best_lasso, f"Tuned Lasso Regression (Degree {best_params_lasso['degree']}, Alpha {best_params_lasso['alpha']})")

# ========================== Model Comparison Table ==========================
results = pd.DataFrame({
    'Model': ['Linear', 'Tuned Ridge', 'Tuned Lasso'],
    'CV Mean R²': [np.max(cv_scores_linear), grid_ridge.best_score_, best_score]
})

print("--- Model Comparison ---")
print(results)

# ========================== Bar Chart with Labels ==========================
models = ['Linear', 'Tuned Ridge', 'Tuned Lasso']
scores = [np.max(cv_scores_linear), grid_ridge.best_score_, best_score]

plt.figure(figsize=(8, 5))
bars = plt.bar(models, scores)
plt.ylim(0, 1)
plt.ylabel('Cross-Validation Mean R² Score')
plt.title('Model Comparison: Linear vs Tuned Ridge vs Tuned Lasso')
plt.grid(axis='y')

# Add value labels
for bar, score in zip(bars, scores):
    plt.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.01,
             f"{score:.4f}",
             ha='center', va='bottom', fontsize=10)

plt.savefig('./poly_model_comparison_bar_chart.png', bbox_inches='tight', dpi=300)
plt.show()

# Close the text file
sys.stdout.close()
