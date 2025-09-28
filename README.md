Jet Engine Remaining Useful Life Prediction

This project uses NASA’s C-MAPSS turbofan engine dataset to build predictive models (LSTM/RNN) that estimate the Remaining Useful Life (RUL) of jet engines. The goal is to enable predictive maintenance by identifying how many flight cycles remain before an engine requires servicing, based on multivariate time-series sensor data.

Dataset
	•	Primary Dataset: NASA C-MAPSS Turbofan Engine Data
	•	Alternate sources:
	•	Kaggle Dataset
	•	PHM Society Archive

The dataset contains simulated sensor readings from jet engines under different operating conditions and fault scenarios. The task is framed as a time-series regression problem, where the model predicts RUL from sensor degradation patterns.

Project Structure

N-CMAPSS-Engine-Prediction/
│
├── data/                     # Datasets
│   ├── raw/                  # Original raw data
│   ├── processed/            # Preprocessed/cleaned data
│   └── external/             # Additional/reference data
│
├── src/                      # Source code
│   ├── data/                 # Data loading & preprocessing
│   │   ├── load_data.py
│   │   └── preprocess.py
│   │
│   ├── models/               # Model architectures & training
│   │   ├── lstm_model.py
│   │   ├── rnn_model.py
│   │   └── train.py
│   │
│   ├── utils/                # Utilities
│   │   ├── metrics.py
│   │   ├── visualization.py
│   │   └── config.py
│   │
│   └── evaluate.py           # Evaluation script
│
├── main.py                   # Main entry point to run pipeline
├── requirements.txt          # Dependencies
├── uv.lock                   # uv dependency lockfile
├── pyproject.toml            # uv project config
├── README.md                 # Project documentation
└── .gitignore                # Ignore datasets, logs, etc.

Setup

This project uses uv for environment and dependency management (instead of pip or conda).

Installation

# Install uv if not already installed
pip install uv

# Sync environment
uv sync

Running the project

python main.py

Next Steps
	•	Implement preprocessing pipeline (scaling, windowing, sequence creation).
	•	Develop LSTM and RNN architectures.
	•	Train models and evaluate using RMSE and other scoring functions.

⸻
