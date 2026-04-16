import numpy as np

def inverse_transform_predictions(predictions, scaler, n_features=3):
    if predictions.ndim == 1:
        predictions = predictions.reshape(-1, 1)
    inv_matrix = np.zeros((len(predictions), n_features))
    inv_matrix[:, -1] = predictions.flatten()
    inv_yhat = scaler.inverse_transform(inv_matrix)[:, -1]
    return inv_yhat

def mean_absolute_percentage_error_safe(y_true, y_pred):
    # Replace zeros in y_true to avoid division by zero (epsilon safety logic)
    y_true_safe = np.where(y_true == 0, 1e-10, y_true)
    return np.mean(np.abs((y_true_safe - y_pred) / y_true_safe)) * 100

def evaluate_metrics(y_true, y_pred):
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    mape = mean_absolute_percentage_error_safe(y_true, y_pred)
    return {'RMSE': rmse, 'MAE': mae, 'MAPE': mape}
