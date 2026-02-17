# NVIDIA Stock Price Prediction Model

A comprehensive machine learning project for predicting NVIDIA (NVDA) stock prices using multiple time series forecasting models including ARIMA, LSTM, GRU, CNN-BiGRU, Transformer, and Ensemble methods.

## Overview

This project implements and compares various machine learning approaches for stock price prediction:
- **Statistical Models**: ARIMA (AutoRegressive Integrated Moving Average)
- **Deep Learning Models**: 
  - LSTM (Long Short-Term Memory)
  - GRU (Gated Recurrent Unit)
  - CNN-BiGRU (Convolutional Neural Network with Bidirectional GRU)
  - Transformer with Positional Encoding
- **Ensemble Methods**: Simple and weighted averaging of predictions

The models use technical indicators (RSI, MACD, Bollinger Bands, etc.) and market volatility data (VIX) as features to predict future stock prices.

## Features

- **Data Sources**:
  - Yahoo Finance for historical NVDA stock data and VIX (volatility index)
  - HuggingFace MTBench dataset for high-frequency trading data
  - FNSPID dataset for additional financial data
  
- **Technical Indicators**:
  - RSI (Relative Strength Index)
  - MACD (Moving Average Convergence Divergence)
  - Bollinger Bands
  - OBV (On-Balance Volume)
  - Volume Weighted Average Price (VWAP)
  
- **Model Evaluation Metrics**:
  - RMSE (Root Mean Squared Error)
  - MAE (Mean Absolute Error)
  - MAPE (Mean Absolute Percentage Error)
  - Directional Accuracy

## Project Structure

```
PredictionStockModel/
├── StockModelTrainBuilding.ipynb  # Main notebook with all models
├── data/
│   ├── raw/                       # Raw data from sources
│   └── processed/                 # Processed features and model outputs
├── models/                        # Saved trained models
├── requirements.txt               # Python dependencies
├── README.md                      # This file
├── LICENSE                        # MIT License
└── .gitignore                     # Git ignore rules
```

## Requirements

- Python 3.8+
- TensorFlow 2.15+
- pandas 2.0+
- See `requirements.txt` for complete list

## Installation

1. Clone the repository:
```bash
git clone https://github.com/DuvanRCuero/PredictionStockModel.git
cd PredictionStockModel
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Open the Jupyter notebook:
```bash
jupyter notebook StockModelTrainBuilding.ipynb
```

2. Run all cells sequentially to:
   - Download and prepare data
   - Engineer features
   - Train multiple models
   - Evaluate and compare results
   - Generate visualizations

3. The notebook will create:
   - `data/raw/` and `data/processed/` directories with datasets
   - `models/` directory with trained model files
   - Visualizations of predictions and model performance

## Models

### ARIMA
Traditional statistical approach for time series forecasting. Auto-optimizes parameters (p, d, q) using AIC.

### LSTM
Deep learning model with Long Short-Term Memory cells, effective for capturing long-term dependencies in sequential data.

### GRU
Simplified version of LSTM with fewer parameters, often training faster while maintaining performance.

### CNN-BiGRU
Combines Convolutional layers for local pattern extraction with Bidirectional GRU for temporal modeling.

### Transformer
Attention-based architecture with positional encoding to capture temporal relationships in time series data.

### Ensemble
Combines predictions from multiple models using simple and weighted averaging for improved robustness.

## Results

The notebook provides comprehensive comparison of all models on:
- Validation and test set performance
- Prediction vs actual price visualizations
- Metric tables comparing RMSE, MAE, MAPE, and directional accuracy

## Data Privacy

This project uses publicly available data sources. All data is stored locally and not shared.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Data sources: Yahoo Finance, HuggingFace, FNSPID
- Libraries: TensorFlow, scikit-learn, pandas, yfinance, ta, pmdarima

## Disclaimer

This project is for educational and research purposes only. It should not be used as financial advice for actual trading decisions. Stock market predictions are inherently uncertain, and past performance does not guarantee future results.
