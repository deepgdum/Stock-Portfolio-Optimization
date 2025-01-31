import numpy as np
import pandas as pd
import yfinance as yf
import cvxpy as cp
import streamlit as st
import plotly.express as px
from datetime import datetime, timedelta

# 1. Fetch Real-Time Stock Data with Error Handling
def get_stock_data(tickers, start, end):
    try:
        data = yf.download(tickers, start=start, end=end)
        
        # Check if 'Adj Close' column exists in the data
        if 'open' not in data.columns:
            data = data.open
        else:
            st.error("'open' column not found in the downloaded data.")
            return None, None
        
        
        # Check if data is empty
        if data.empty:
            st.error("No data found for the given tickers and date range.")
            return None, None
        
        returns = data['open'].pct_change().dropna()
        return data, returns
    
    except Exception as e:
        st.error(f"Error fetching stock data: {e}")
        return None, None

# 2. Calculate Expected Returns & Covariance Matrix
def calculate_metrics(returns):
    if returns is None or returns.empty:
        st.error("Returns data is empty!")
        return None, None
    
    # Calculate expected returns and covariance matrix
    mu = returns.mean()  # Expected returns
    sigma = returns.cov()  # Covariance matrix
    
    # Debugging print statements
    st.write(f"Expected returns (mu): {mu}")
    st.write(f"Covariance matrix (sigma): {sigma}")
    
    # Check if mu and sigma are valid
    if mu.isnull().any() or sigma.isnull().any().any():
        st.error("There are NaN values in the expected returns or covariance matrix!")
        return None, None

    return mu, sigma

# 3. Portfolio Optimization with Error Handling
def optimize_portfolio(mu, sigma, prev_weights=None, risk_aversion=0.5, short_selling=False, transaction_cost=0.001):
    if mu is None or sigma is None:
        st.error("Expected returns (mu) or covariance matrix (sigma) is None!")
        return None
    
    try:
        n = len(mu)

        # Check that mu is a 1D array (vector) and sigma is a 2D square matrix
        if len(mu.shape) != 1:
            st.error(f"Expected returns (mu) should be a 1D array, but got shape {mu.shape}.")
            return None

        if len(sigma.shape) != 2 or sigma.shape[0] != sigma.shape[1] or sigma.shape[0] != n:
            st.error(f"Covariance matrix (sigma) should be square with shape (n, n), but got shape {sigma.shape}.")
            return None
        
        # Debugging print statements
        st.write(f"Optimization problem with {n} assets.")
        
        # Define portfolio weights variable
        w = cp.Variable(n)
        
        # Define the objective function and constraints
        objective = cp.Maximize(mu.T @ w - risk_aversion * cp.quad_form(w, sigma))
        constraints = [cp.sum(w) == 1]
        
        if not short_selling:
            constraints.append(w >= 0)
        
        if prev_weights is not None:
            trade_cost = cp.norm(w - prev_weights, 1) * transaction_cost
            objective -= trade_cost
        
        # Define and solve the problem
        problem = cp.Problem(objective, constraints)
        problem.solve()

        # Debugging: check the value of w
        if w.value is None:
            st.error(f"Optimization failed. The value of portfolio weights (w) is None.")
            return None

        st.write(f"Optimal portfolio weights: {w.value}")
        return w.value

    except Exception as e:
        st.error(f"Optimization error: {e}")
        return None

# 4. Risk-Adjusted Metrics (Sharpe Ratio)
def calculate_sharpe_ratio(portfolio_returns, risk_free_rate=0.02):
    if portfolio_returns is None or portfolio_returns.empty:
        return None
    excess_returns = portfolio_returns - risk_free_rate
    return np.mean(excess_returns) / np.std(excess_returns) if np.std(excess_returns) > 0 else None

# 5. Trigger-Based Rebalancing with Safe Handling
def trigger_rebalancing(tickers, start, end, initial_capital=100000, risk_aversion=0.5, transaction_cost=0.001, weight_deviation_threshold=0.05):
    data, returns = get_stock_data(tickers, start, end)
    if data is None or returns is None:
        return None, None, None, None, None
    
    mu, sigma = calculate_metrics(returns)
    if mu is None or sigma is None:
        return None, None, None, None, None
    
    dates = data.index
    portfolio_weights = {}
    prev_weights = None
    portfolio_values = []
    capital = initial_capital
    transaction_costs = []
    rebalance_dates = []
    
    for date in dates:
        if prev_weights is None:
            opt_weights = optimize_portfolio(mu, sigma, prev_weights, risk_aversion, transaction_cost=transaction_cost)
            if opt_weights is None:
                return None, None, None, None, None
            portfolio_weights[date] = opt_weights
            prev_weights = opt_weights
            rebalance_dates.append(date)
        else:
            if date not in returns.index:
                continue
            
            current_weights = prev_weights * (1 + returns.loc[date])
            current_weights /= np.sum(current_weights)
            
            deviation = np.abs(current_weights - prev_weights)
            if np.any(deviation > weight_deviation_threshold):
                opt_weights = optimize_portfolio(mu, sigma, prev_weights, risk_aversion, transaction_cost=transaction_cost)
                if opt_weights is None:
                    return None, None, None, None, None
                portfolio_weights[date] = opt_weights
                prev_weights = opt_weights
                rebalance_dates.append(date)
            else:
                portfolio_weights[date] = current_weights
                prev_weights = current_weights
        
        capital *= (1 + np.dot(prev_weights, returns.loc[date]))
        portfolio_values.append(capital)
        
        if len(rebalance_dates) > 1 and date == rebalance_dates[-1]:
            trade_cost = np.sum(np.abs(portfolio_weights[date] - portfolio_weights[rebalance_dates[-2]])) * transaction_cost
            transaction_costs.append(trade_cost)
        else:
            transaction_costs.append(0)
    
    portfolio_returns = pd.Series(portfolio_values).pct_change().dropna()
    sharpe_ratio = calculate_sharpe_ratio(portfolio_returns)
    
    return portfolio_weights, portfolio_values, transaction_costs, sharpe_ratio, rebalance_dates

# 6. Visualization Dashboard with None Checks
def portfolio_dashboard(portfolio_values, dates, transaction_costs, sharpe_ratio, rebalance_dates):
    st.title("📈 Trigger-Based Portfolio Optimization Dashboard")
    
    if portfolio_values is None or len(portfolio_values) == 0:
        st.error("No portfolio values available for visualization.")
        return
    
    # Portfolio Performance Chart
    df = pd.DataFrame({"Date": dates, "Portfolio Value": portfolio_values})
    fig1 = px.line(df, x="Date", y="Portfolio Value", title="Portfolio Performance Over Time", labels={"Portfolio Value": "Portfolio Value ($)"})
    fig1.update_layout(showlegend=False, xaxis_title="Date", yaxis_title="Portfolio Value ($)", template="plotly_dark")
    st.plotly_chart(fig1)
    
    # Transaction Costs Chart
    df_costs = pd.DataFrame({"Date": dates, "Transaction Cost": transaction_costs})
    fig2 = px.bar(df_costs, x="Date", y="Transaction Cost", title="Transaction Costs Over Time", labels={"Transaction Cost": "Transaction Cost ($)"})
    fig2.update_layout(xaxis_title="Date", yaxis_title="Transaction Cost ($)", template="plotly_dark")
    st.plotly_chart(fig2)
    
    # Rebalancing Events Chart
    df_rebalance = pd.DataFrame({"Date": rebalance_dates, "Event": ["Rebalance"] * len(rebalance_dates)})
    fig3 = px.scatter(df_rebalance, x="Date", y=[0] * len(rebalance_dates), title="Rebalancing Events", labels={"Date": "Rebalance Date"})
    fig3.update_layout(showlegend=False,xaxis_title="Date", yaxis_title="Event", template="plotly_dark")
    st.plotly_chart(fig3)
    
    # Sharpe Ratio
    if sharpe_ratio is not None:
        st.metric("📊 Sharpe Ratio", f"{sharpe_ratio:.2f}")
    else:
        st.error("Unable to calculate Sharpe Ratio.")
# 7. Main Function with Safe Execution
def main():
    st.sidebar.title("Portfolio Optimization Parameters")
    tickers = st.sidebar.text_input("Enter Tickers (comma-separated)", "AAPL,MSFT,GOOGL,TSLA,AMZN").split(',')
    start_date = st.sidebar.date_input("Start Date", datetime.now() - timedelta(days=365))
    end_date = st.sidebar.date_input("End Date", datetime.now())
    risk_aversion = st.sidebar.slider("Risk Aversion", 0.0, 1.0, 0.5, 0.01)
    
    st.sidebar.info("🔧 Adjust settings and press 'Run Optimization' to trigger portfolio analysis.")
    
    if st.sidebar.button("Run Optimization"):
        portfolio, portfolio_values, transaction_costs, sharpe_ratio, rebalance_dates = trigger_rebalancing(
            tickers, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), risk_aversion=risk_aversion
        )
        
        if portfolio_values:
            dates = pd.date_range(start_date, end_date, freq='D')[:len(portfolio_values)]
            portfolio_dashboard(portfolio_values, dates, transaction_costs, sharpe_ratio, rebalance_dates)

if __name__ == "__main__":
    main()
