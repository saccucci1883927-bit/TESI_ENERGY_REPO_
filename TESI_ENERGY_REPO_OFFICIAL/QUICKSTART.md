# QUICKSTART Guide

## Quick Setup Guide

1. **Clone the Repository**  
   Run the following command:
   ```bash
   git clone https://github.com/TuoUsername/Tesi_Energy_Forecasting.git
   cd Tesi_Energy_Forecasting
   ```

2. **Install Dependencies**  
   For setup, install python modules (tensorflow, pandas, sk-learn):
   ```bash
   pip install -r requirements.txt
   ```

3. **Add Dataset**  
   Place your dataset `.csv` within the `data/` folder and name it `dataset_finale_4_anni.csv`.

4. **Run the Model Training**  
   We configured the fastest production-ready pipeline script for cross-platform usage:
   ```bash
   python notebooks/01_Model_Training_REFACTORED.py
   ```

5. **View Results**  
   After running the script, metrics output is dynamically saved into `results/metrics_comparison.csv`. Saved Models (the generated .h5 arrays) will populate inside `models_saved/`.
