from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# ===== Evaluation =====
def evaluate(y_true, y_pred, model_name):
    print(f"--- {model_name} ---")
    print("RMSE:", np.sqrt(mean_squared_error(y_true, y_pred)))
    print("MAE:", mean_absolute_error(y_true, y_pred))
    print("R2 Score:", r2_score(y_true, y_pred))
    print()