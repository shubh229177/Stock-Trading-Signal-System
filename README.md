# 📈 Algorithmic Trading System using Machine Learning

## Overview

This project is a complete **algorithmic trading pipeline** built using Python. It combines **technical indicators**, **machine learning models**, and **rule-based logic** to generate trading signals (BUY/SELL) and simulate real trading performance through backtesting.

The goal of this system is to:

* Predict short-term price movements
* Generate reliable trading signals
* Combine indicator-based and ML-based decisions
* Evaluate profitability using a simulated trading strategy

---

## Data Collection

* Historical stock data is fetched using **yfinance**
* Asset used: **AAPL (Apple Inc.)**
* Time period: **2020 to April 2026**

```python
data = yf.download("AAPL", start='2020-01-01', end='2026-04-18')
```

---

## Feature Engineering

A wide range of **technical indicators** are created to capture market behavior:

### Price-Based Features

* `Price_Change`
* `pct_change`

### RSI Components

* Gain / Loss separation
* Average Gain / Loss
* Relative Strength (RS)
* RSI calculation

### Moving Averages

* MA_10, MA_20, MA_50, MA_100
* MA Ratio (trend strength)

### Volatility & Momentum

* Rolling standard deviation (Volatility)
* Momentum (5-day & 10-day)
* Rate of Change (ROC)

### Trend Indicators

* Exponential Moving Average (EMA_10)
* MACD

### Bollinger Bands

* Upper Band
* Lower Band
* Band Width (volatility measure)

### Volume Features

* Volume moving average
* Volume ratio
* Volume change

### Lag Features

* RSI Lag (1,2)
* Return Lag (1,2)

---

## Rule-Based Signal Generation

Initial signals are generated using **RSI thresholds**:

* BUY → RSI crosses above 30
* SELL → RSI crosses below 70
* HOLD → otherwise

```python
data.loc[(data["RSI"] < 30) & (data["RSI"].shift(1) >= 30), "Signal"] = "BUY"
data.loc[(data["RSI"] > 70) & (data["RSI"].shift(1) <= 70), "Signal"] = "SELL"
```

---

## Target Variable Creation

Future returns are used to define classification labels:

* BUY → Future return > 2%
* SELL → Future return < -2.5%
* HOLD → ignored

```python
data['Future_Return'] = (data['Close'].shift(-3) - data['Close']) / data['Close']
```

This ensures the model learns **forward-looking patterns**.

---

## Machine Learning Models Used

### 1. Logistic Regression

* Simple baseline model
* Fast and interpretable
* Used to check linear separability

### 2. Random Forest

* Ensemble of decision trees
* Captures non-linear relationships
* Reduces overfitting compared to single trees

### 3. XGBoost (Final Model)

* Gradient boosting algorithm
* Handles complex patterns efficiently
* Best performing model in this project

```python
model_XG = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8
)
```

---

## Time Series Cross Validation

Instead of random split, **TimeSeriesSplit** is used:

* Maintains chronological order
* Prevents data leakage
* More realistic for trading systems

---

## Data Scaling

* StandardScaler is applied
* Required for models like Logistic Regression
* Ensures features are normalized

---

## Class Imbalance Handling

* Class weights computed using:

```python
compute_class_weight('balanced', ...)
```

* Helps model not bias towards majority class

---

## Model Performance

| Model              | Accuracy |
| ------------------ | -------- |
| LogisticRegression | ~50%     |
| RandomForest       | ~50%     |
| XGBoost            | ~51%     |

### Key Insight:

* Accuracy is low but acceptable in trading
* Even slight edge + good risk management = profit

---

## Signal Fusion (Hybrid Strategy)

Final signal combines:

* RSI-based signal
* ML prediction

### Logic:

| Condition    | Final Signal |
| ------------ | ------------ |
| Both BUY     | STRONG BUY   |
| Both SELL    | STRONG SELL  |
| Only ML BUY  | WEAK BUY     |
| Only ML SELL | WEAK SELL    |
| Else         | HOLD         |

---

## Trading Strategy

### Entry Rules

* Buy only if:

  * Signal is BUY
  * Trend is UP (MA50 > MA100)

### Exit Rules

* Sell on SELL signal

### Risk Management

* Stop Loss: 3%
* Take Profit: 6%

---

## Backtesting Results

* Initial Capital: 10,000
* Final Capital: 11,694
* Profit: 1,694
* Return: ~16.7%
* Total Trades: 8
* Win Rate: 62.5%

---

## Visualization

* Matplotlib: Static plots
* Plotly: Interactive charts
* BUY/SELL points plotted on price chart

---

## Confidence System

A simple confidence scoring:

* 2 → Strong agreement (RSI + ML)
* 1 → Only ML
* 0 → Weak signal

---

## Output Files

* `final_output.csv` → processed dataset
* `dashboard.json` → for dashboard integration

---

## Key Learnings

1. Technical indicators alone are not enough
2. ML improves signal quality slightly
3. Combining both gives better decisions
4. Risk management is more important than accuracy
5. Time series validation is critical

---

## Limitations

* No transaction costs included properly
* No slippage modeling
* Overfitting risk (XGBoost train score = 1.0)
* Limited feature selection tuning

---

## Future Improvements

* Hyperparameter tuning (GridSearch / Optuna)
* Add more indicators (ATR, ADX)
* Use LSTM / Deep Learning
* Add portfolio management
* Real-time deployment with API
* Include transaction cost + slippage

---

## Conclusion

This project demonstrates a **complete end-to-end trading system**:

* Data → Features → ML → Signals → Backtesting

Even with moderate accuracy, the system achieves **profitable results** due to:

* Strong feature engineering
* Hybrid signal logic
* Proper risk management

---

## Author

Shubham Thakur

---
