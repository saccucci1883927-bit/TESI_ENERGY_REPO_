# SETUP.md

## Installation Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/TuoUsername/Tesi_Energy_Forecasting.git
   cd Tesi_Energy_Forecasting
   ```

2. **Install dependencies**
   For Python projects, create a virtual environment and install requirements:
   ```bash
   python -m venv venv
   source venv/bin/activate  # In Windows usa `venv\Scripts\activate`
   pip install -r requirements.txt
   ```

## Project Structure
```text
Tesi_Energy_Forecasting/
├── src/                # Moduli Python (preprocessing.py, architectures.py, evaluation.py)
├── notebooks/          # Script e Notebooks (01_Model_Training_REFACTORED.py)
├── data/               # CSV dati energetici
├── requirements.txt    # Librerie necessarie
├── CONFIG.py           # Configurazione centralizzata parametri HyperTuning
└── README.md           # Riepilogo del Progetto Tesi
```

## Configuration Guide
Modifica il file `CONFIG.py` situato nella directory principale per adattare learning rate, batch size, epochs o percorsi personalizzati.

## Troubleshooting
**Common Issues**:
* **ModuleNotFoundError**: Assicurati di attivare l'ambiente virtuale (`venv`) e fare `pip install -r requirements.txt`.
* **Dataset Not Found**: Verifica di aver posto *dataset_finale_4_anni.csv* all'interno di `data/`.
