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

## 📊 Results & Performance

For detailed performance analysis, see **[RESULTS_SHOWCASE.md](RESULTS_SHOWCASE.md)**.

### Quick Summary

| Model                    | RMSE ($) | MAPE (%) | Directional Accuracy |
|--------------------------|----------|----------|---------------------|
| **Stacking Ensemble**    | **5.23** | **3.12%** | **58.7%**          |
| Transformer              | 5.89     | 3.67%    | 56.2%               |
| CNN-BiGRU               | 6.12     | 3.89%    | 55.8%               |
| LSTM                     | 6.34     | 4.02%    | 55.1%               |
| GRU                      | 6.41     | 4.08%    | 54.9%               |
| ARIMA                    | 8.92     | 5.67%    | 52.3%               |
| Naive Baseline          | 12.45    | 7.83%    | 50.0%               |

### Key Achievements
- ✅ **58% better RMSE** than naive baseline
- ✅ **17% improvement** in directional accuracy over random
- ✅ **Competitive with published research** (3.12% MAPE)
- ✅ **Eliminated feature leakage** for realistic estimates
- ✅ **Stacking ensemble** shows 5-15% improvement over single models

---

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
├── StockModelTrainBuilding.ipynb     # Main notebook with all models
├── RESULTS_SHOWCASE.md                # Detailed performance analysis
├── ARCHITECTURAL_IMPROVEMENTS.md      # Documentation of improvements
├── generate_visualizations.py         # Script to create charts
├── data/
│   ├── raw/                          # Raw data from sources
│   └── processed/                    # Processed features and model outputs
├── models/                           # Saved trained models
├── visualizations/                   # Generated performance charts
├── requirements.txt                  # Python dependencies
├── README.md                         # This file
├── LICENSE                           # MIT License
└── .gitignore                        # Git ignore rules
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

4. (Optional) Generate additional visualizations:
```bash
python generate_visualizations.py
```

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

### Stacking Ensemble
Combines predictions from multiple models using a Ridge regression meta-learner for optimal weighting.

## Data Privacy

This project uses publicly available data sources. All data is stored locally and not shared.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Data sources: Yahoo Finance, HuggingFace, FNSPID
- Libraries: TensorFlow, scikit-learn, pandas, yfinance, ta, pmdarima
- Architectural improvements documented in [ARCHITECTURAL_IMPROVEMENTS.md](ARCHITECTURAL_IMPROVEMENTS.md)

## Disclaimer

This project is for educational and research purposes only. It should not be used as financial advice for actual trading decisions. Stock market predictions are inherently uncertain, and past performance does not guarantee future results.
