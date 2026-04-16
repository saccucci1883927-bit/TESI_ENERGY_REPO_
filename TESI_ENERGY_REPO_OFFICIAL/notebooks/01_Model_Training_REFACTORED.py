import numpy as np
import pandas as pd
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.preprocessing import get_data_path, load_data, prepare_data_ts, inverse_transform_predictions
from src.architectures import build_lstm, build_bilstm, build_cnn_lstm, build_cnn_bilstm, build_cnn_bilstm_attention
from src.evaluation import evaluate_metrics
from CONFIG import model_params, output_directory, dataset_path

def build_models(N_IN, N_FEATURES):
    return {
        'model_lstm': build_lstm(N_IN, N_FEATURES),
        'model_bilstm': build_bilstm(N_IN, N_FEATURES),
        'model_cnn_lstm': build_cnn_lstm(N_IN, N_FEATURES),
        'model_cnn_bilstm': build_cnn_bilstm(N_IN, N_FEATURES),
        'model_cnn_bilstm_attention': build_cnn_bilstm_attention(N_IN, N_FEATURES)
    }

def train_model(model, train_X, train_y, val_X, val_y, epochs, batch_size):
    model.fit(
        train_X, train_y, 
        validation_data=(val_X, val_y), 
        epochs=epochs, 
        batch_size=batch_size, 
        verbose=1, 
        shuffle=False
    )
    return model

def main():
    print("Avvio Pipeline di Addestramento Refactored...")
    
    full_dataset_path = get_data_path(dataset_path)
    try:
        data = load_data(full_dataset_path)
    except Exception as e:
        print(f"Errore caricamento dati: {e}")
        return
        
    values = data.values
    N_IN, N_FEATURES = 72, 3
    
    print("Preparazione dei dati (Split Rigoroso No-Leakage)...")
    train_X, train_y, val_X, val_y, test_X, test_y, scaler = prepare_data_ts(
        values, n_in=N_IN, n_out=1, n_features=N_FEATURES, test_split=0.15, val_split=0.1
    )
    
    models = build_models(N_IN, N_FEATURES)
    results = {}
    
    os.makedirs(os.path.join(os.path.dirname(__file__), f"../{output_directory}"), exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(__file__), '../models_saved'), exist_ok=True)
    
    for name, model in models.items():
        print(f"\n--- Training {name} ---")
        trained_model = train_model(
            model, train_X, train_y, val_X, val_y, 
            epochs=model_params['num_epochs'], 
            batch_size=model_params['batch_size']
        )
        
        # Salvataggio
        model_path = os.path.join(os.path.dirname(__file__), f'../models_saved/{name}.h5')
        trained_model.save(model_path)
        
        # Predizione Valutativa sul Test Set
        test_predictions = trained_model.predict(test_X)
        inv_test_preds = inverse_transform_predictions(test_predictions, scaler, n_features=N_FEATURES)
        inv_test_y = inverse_transform_predictions(test_y, scaler, n_features=N_FEATURES)

        # Calcolo Metriche
        metrics = evaluate_metrics(inv_test_y, inv_test_preds)
        results[name] = metrics
        
        print(f"Metriche {name}: {metrics}")

    # Export risultati finali
    results_df = pd.DataFrame(results).T
    csv_path = os.path.join(os.path.dirname(__file__), f"../{output_directory}metrics_comparison.csv")
    results_df.to_csv(csv_path)
    print(f"\nFine procedura. I risultati sono stati esportati in: {csv_path}")

if __name__ == "__main__":
    main()
