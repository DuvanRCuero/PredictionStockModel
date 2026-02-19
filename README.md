## 📊 Results & Performance

For detailed performance analysis, see **[RESULTS_SHOWCASE.md](RESULTS_SHOWCASE.md)**.

### Quick Summary

| Model                    | RMSE ($) | MAPE (%) | Directional Accuracy |
|--------------------------|----------|----------|---------------------|
| **Stacking Ensemble**    | 5.23     | 3.12%    | 58.7%               |
| Transformer              | 5.89     | 3.67%    | 56.2%               |
| CNN-BiGRU               | 6.12     | 3.89%    | 55.8%               |
| LSTM                     | 6.34     | 4.02%    | 55.1%               |
| GRU                      | 6.41     | 4.08%    | 54.9%               |
| ARIMA                    | 8.92     | 5.67%    | 52.3%               |
| Naive Baseline          | 12.45    | 7.83%    | 50.0%               |

<!-- ![Model Performance](visualizations/model_performance_comparison.png) -->

### Key Achievements
- ✅ 58% better RMSE than naive baseline
- ✅ 17% improvement in directional accuracy over random
- ✅ Competitive with published research (3.12% MAPE)
- ✅ Eliminated feature leakage for realistic estimates

## Model Comparison Details

