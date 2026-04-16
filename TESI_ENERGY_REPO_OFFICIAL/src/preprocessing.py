import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler

def get_data_path(relative_path):
    """Returns a cross-platform path to the data file."""
    # Since this file is in src/, the root directory is one level up
    root_dir = os.path.dirname(os.path.dirname(__file__))
    return os.path.join(root_dir, relative_path)

def load_data(file_path):
    """Loads data from the specified path with error handling."""
    try:
        data = pd.read_csv(file_path, header=0, index_col=0, decimal=',', thousands='.')
        return data
    except FileNotFoundError:
        raise Exception(f"File not found: {file_path}")
    except pd.errors.EmptyDataError:
        raise Exception(f"No data: {file_path}")
    except Exception as e:
        raise Exception(f"An error occurred while loading data: {str(e)}")

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    if dropnan:
        agg.dropna(inplace=True)
    return agg

def prepare_data_ts(values, n_in=72, n_out=1, n_features=3, test_split=0.15, val_split=0.1):
    """
    Ripartizione temporale logica per evitare data leakage in Time Series:
    Train | Val | Test
    """
    total_samples = len(values)
    
    test_idx = int(total_samples * (1 - test_split))
    val_idx = int(test_idx * (1 - val_split))
    
    train_values = values[:val_idx, :]
    val_values = values[val_idx:test_idx, :]
    test_values = values[test_idx:, :]
    
    scaler = RobustScaler()
    scaled_train = scaler.fit_transform(train_values)
    scaled_val = scaler.transform(val_values)
    scaled_test = scaler.transform(test_values)
    
    reframed_train = series_to_supervised(scaled_train, n_in=n_in, n_out=n_out)
    reframed_val = series_to_supervised(scaled_val, n_in=n_in, n_out=n_out)
    reframed_test = series_to_supervised(scaled_test, n_in=n_in, n_out=n_out)
    
    n_input_cols = n_in * n_features
    
    def extract_X_y(reframed):
        vals = reframed.values
        X, y_full = vals[:, :n_input_cols], vals[:, n_input_cols:]
        y = y_full[:, n_features-1::n_features]
        X = X.reshape((X.shape[0], n_in, n_features))
        return X, y

    train_X, train_y = extract_X_y(reframed_train)
    val_X, val_y = extract_X_y(reframed_val)
    test_X, test_y = extract_X_y(reframed_test)
    
    return train_X, train_y, val_X, val_y, test_X, test_y, scaler

def inverse_transform_predictions(predictions, scaler, n_features=3):
    """Inverse transforms the predictions from scaled back to original values."""
    if predictions.ndim == 1:
        predictions = predictions.reshape(-1, 1)
    inv_matrix = np.zeros((len(predictions), n_features))
    inv_matrix[:, -1] = predictions.flatten()
    inv_yhat = scaler.inverse_transform(inv_matrix)[:, -1]
    return inv_yhat
