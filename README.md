# 📈 Adaptive Portfolio Optimization Dashboard

An interactive, real-time portfolio optimization dashboard built with **Python** and **Streamlit**, using **Modern Portfolio Theory (MPT)**, dynamic rebalancing, and customizable user preferences like risk aversion and short-selling.

---

## 🚀 Project Need

In financial markets, maintaining an optimal asset allocation is crucial for maximizing returns while managing risk. Traditional buy-and-hold strategies often fail to adapt to market fluctuations. This project addresses the need for:

- **Real-time portfolio optimization**
- **Dynamic asset rebalancing based on volatility and drift**
- **Custom control over risk and constraints**
- **Interactive visualization for better decision-making**

The dashboard empowers users to experiment with asset combinations, risk levels, and rebalancing strategies — all through an easy-to-use UI.

---

## 🎯 Features

- 📊 Live asset data pulled via `yfinance`
- 💼 Modern Portfolio Theory-based optimization with `cvxpy`
- 🔁 Automatic rebalancing using user-defined thresholds
- 🔒 Optional short-selling and transaction cost settings
- 📈 Real-time visualization with `Plotly`
- 🌐 Lightweight web interface built with `Streamlit`

---

## 🛠️ Tech Stack

- **Python**
- **Streamlit** – Web UI
- **yFinance** – Financial data API
- **cvxpy** – Portfolio optimization modeling
- **Pandas, NumPy** – Data manipulation
- **Plotly** – Interactive plots

---

## ⚙️ How It Works – Function-by-Function Breakdown

### 1. `get_data(tickers, start_date, end_date)`
Fetches historical adjusted close prices from Yahoo Finance for the given list of tickers.

- Returns a cleaned `pandas.DataFrame` with prices aligned by date.
- Handles missing data using forward and backward fill.

---

### 2. `calculate_portfolio_value(weights, prices)`
Calculates the total portfolio value over time based on the weighted sum of asset prices.

- Uses dot product of weights and normalized price series.
- Normalizes results to start from 1 for comparability.

---

### 3. `calculate_statistics(weights, returns, risk_aversion, penalty)`
Computes the portfolio’s **expected return**, **volatility**, and **Sharpe ratio**.

- Adjusts for risk preferences via the `risk_aversion` parameter.
- Applies penalty for turnover to simulate transaction costs.

---

### 4. `optimize_portfolio(returns, allow_short, risk_aversion, penalty)`
Builds and solves a convex optimization problem using **cvxpy**.

- Objective: Maximize risk-adjusted return (Sharpe ratio).
- Constraints:
  - Sum of weights = 1
  - Optional: No short-selling
- Returns the optimized weight vector.

---

### 5. `rebalance_portfolio(current_weights, new_weights, threshold)`
Checks if the deviation between current and optimal weights exceeds the threshold.

- If true: triggers a rebalance and updates portfolio.
- Helps reduce unnecessary trades and costs.

---

### 6. `simulate_portfolio(...)`
Main loop to simulate portfolio behavior over time.

- Rebalances at defined intervals or thresholds.
- Applies optimization at each step.
- Tracks and stores portfolio value, rebalancing points, and metrics.

---

### 7. `plot_portfolio_value(...)`
Plots portfolio performance over time using **Plotly**.

- Adds markers to indicate rebalancing events.
- Visualizes drawdowns and value evolution.

---

## 🖥️ How to Run Locally

1. **Clone the repo**  
   ```bash
   git clone https://github.com/yourusername/portfolio-optimization-dashboard.git
   cd portfolio-optimization-dashboard
