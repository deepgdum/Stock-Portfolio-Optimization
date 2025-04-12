import numpy as np
import pandas as pd
import yfinance as yf
import cvxpy as cp
import streamlit as st
import plotly.express as px
from datetime import datetime, timedelta

# 1. Fetch Real-Time Stock Data with Improved Error Handling
# Fixed get_stock_data function to handle Yahoo Finance data structure better
def get_stock_data(tickers, start, end):
    try:
        data = yf.download(tickers, start=start, end=end)
        
        if data.empty:
            st.error("No stock data found. Check the ticker symbols or date range.")
            return None, None
        
        # Handle data structure based on number of tickers
        if isinstance(tickers, list) and len(tickers) > 1:
            # For multiple tickers, we get a multi-level column DataFrame
            if 'Close' in data.columns.levels[0]:
                prices = data['Close']
            else:
                st.error("Unable to find 'Close' price data in the downloaded dataset.")
                return None, None
        else:
            # For a single ticker, we get a simple DataFrame
            if 'Close' in data.columns:
                prices = data['Close']
            else:
                st.error("Unable to find 'Close' price data in the downloaded dataset.")
                return None, None
        
        # Calculate returns
        returns = prices.pct_change().dropna()
        
        if returns.empty:
            st.error("Unable to calculate returns. Data might be insufficient.")
            return None, None
        
        return prices, returns
        
    except Exception as e:
        st.error(f"Error fetching stock data: {str(e)}")
        return None, None

# 2. Calculate Expected Returns & Covariance Matrix with Better Error Handling
def calculate_metrics(returns):
    try:
        if returns is None or returns.empty:
            st.error("Returns data is empty or None.")
            return None, None
        
        # Handle both Series and DataFrame cases
        if isinstance(returns, pd.Series):
            mu = np.array([returns.mean()])
            sigma = np.array([[returns.var()]])
        else:
            mu = returns.mean().values
            sigma = returns.cov().values
        
        # Validate results
        if np.isnan(mu).any() or np.isnan(sigma).any():
            st.error("NaN values detected in expected returns or covariance matrix.")
            return None, None
            
        return mu, sigma
        
    except Exception as e:
        st.error(f"Error calculating metrics: {str(e)}")
        return None, None

# 3. Portfolio Optimization with Robust Error Handling
def optimize_portfolio(mu, sigma, prev_weights=None, risk_aversion=0.5, short_selling=False, transaction_cost=0.001):
    try:
        # Basic validation
        if mu is None or sigma is None:
            st.error("Expected returns or covariance matrix is None.")
            return None
            
        n = len(mu)
        
        # Ensure mu is a vector and sigma is a matrix of correct dimensions
        if n == 0:
            st.error("Empty expected returns vector.")
            return None
            
        if sigma.shape != (n, n):
            st.error(f"Covariance matrix dimensions ({sigma.shape}) don't match expected returns length ({n}).")
            return None
        
        # Define portfolio optimization problem
        w = cp.Variable(n)
        
        # Define the objective function with transaction costs if applicable
        objective = cp.Maximize(mu.T @ w - risk_aversion * cp.quad_form(w, sigma))
        
        if prev_weights is not None:
            transaction_penalty = cp.norm(w - prev_weights, 1) * transaction_cost
            objective = cp.Maximize(mu.T @ w - risk_aversion * cp.quad_form(w, sigma) - transaction_penalty)
        
        # Define constraints
        constraints = [cp.sum(w) == 1]
        if not short_selling:
            constraints.append(w >= 0)
        
        # Solve the problem
        problem = cp.Problem(objective, constraints)
        problem.solve()
        
        # Check solution status
        if problem.status not in ["optimal", "optimal_inaccurate"]:
            st.error(f"Optimization failed with status: {problem.status}")
            return None
            
        return w.value
        
    except Exception as e:
        st.error(f"Optimization error: {str(e)}")
        return None

# 4. Risk-Adjusted Metrics (Sharpe Ratio) with Better Validation
def calculate_sharpe_ratio(portfolio_returns, risk_free_rate=0.02, annualize=True):
    try:
        if portfolio_returns is None or len(portfolio_returns) == 0:
            return None
            
        # Convert to numpy array if needed
        if isinstance(portfolio_returns, pd.Series):
            returns_array = portfolio_returns.values
        else:
            returns_array = np.array(portfolio_returns)
            
        # Calculate excess returns
        excess_returns = returns_array - (risk_free_rate / 252 if annualize else risk_free_rate)
        
        # Calculate Sharpe ratio
        std_dev = np.std(excess_returns)
        if std_dev <= 0:
            return 0  # Return 0 instead of None for zero risk
            
        sharpe = np.mean(excess_returns) / std_dev
        
        # Annualize if requested (assuming daily returns)
        if annualize:
            sharpe = sharpe * np.sqrt(252)
            
        return sharpe
        
    except Exception as e:
        st.error(f"Error calculating Sharpe ratio: {str(e)}")
        return None

# 5. Improved Trigger-Based Rebalancing
def trigger_rebalancing(tickers, start, end, initial_capital=100000, risk_aversion=0.5, 
                       transaction_cost=0.001, weight_deviation_threshold=0.05):
    try:
        # Get data
        prices, returns = get_stock_data(tickers, start, end)
        if prices is None or returns is None:
            return None
        
        # Calculate metrics for optimization
        mu, sigma = calculate_metrics(returns)
        if mu is None or sigma is None:
            return None
        
        # Initialize portfolios and tracking variables
        dates = returns.index  # Use returns.index to ensure dates align
        portfolio_weights = {}
        portfolio_values = [initial_capital]  # Start with initial capital
        current_weights = None
        transaction_costs_history = [0]  # No initial transaction cost
        rebalance_dates = []
        current_value = initial_capital
        
        # First allocation
        first_date = dates[0]
        current_weights = optimize_portfolio(mu, sigma, None, risk_aversion, transaction_cost=transaction_cost)
        if current_weights is None:
            return None
            
        portfolio_weights[first_date] = current_weights
        rebalance_dates.append(first_date)
        
        # Process each subsequent date
        for i in range(1, len(dates)):
            date = dates[i]
            prev_date = dates[i-1]
            
            # Get daily returns
            daily_returns = returns.loc[date]
            
            # Calculate new weights after market movements
            if isinstance(daily_returns, pd.Series):
                # Multiple assets
                new_weights = current_weights * (1 + daily_returns.values)
            else:
                # Single asset
                new_weights = current_weights * (1 + daily_returns)
                
            # Normalize weights to sum to 1
            new_weights = new_weights / np.sum(new_weights)
            
            # Check if rebalancing is needed
            max_deviation = np.max(np.abs(new_weights - current_weights))
            
            # Calculate daily portfolio return (before potential rebalancing)
            if isinstance(daily_returns, pd.Series):
                portfolio_return = np.sum(current_weights * daily_returns.values)
            else:
                portfolio_return = current_weights[0] * daily_returns
                
            # Update portfolio value before transaction costs
            current_value *= (1 + portfolio_return)
            
            # Apply rebalancing if threshold exceeded
            transaction_cost_amount = 0
            if max_deviation > weight_deviation_threshold:
                # Optimize for new weights
                optimal_weights = optimize_portfolio(
                    mu, sigma, current_weights, risk_aversion, transaction_cost=transaction_cost
                )
                
                if optimal_weights is not None:
                    # Calculate transaction costs
                    transaction_cost_amount = np.sum(np.abs(optimal_weights - new_weights)) * transaction_cost * current_value
                    
                    # Update current weights and record rebalancing
                    current_weights = optimal_weights
                    rebalance_dates.append(date)
                    
                    # Apply transaction costs to portfolio value
                    current_value -= transaction_cost_amount
                else:
                    # If optimization fails, keep the market-adjusted weights
                    current_weights = new_weights
            else:
                # No rebalancing needed, keep the market-adjusted weights
                current_weights = new_weights
                
            # Record updated portfolio value and weights
            portfolio_values.append(current_value)
            transaction_costs_history.append(transaction_cost_amount)
            portfolio_weights[date] = current_weights.copy()  # Store a copy to avoid reference issues
        
        # Calculate portfolio returns for performance metrics
        portfolio_return_series = pd.Series(portfolio_values).pct_change().dropna()
        sharpe_ratio = calculate_sharpe_ratio(portfolio_return_series)
        
        # Return all necessary data for visualization and analysis
        return {
            'weights': portfolio_weights,
            'values': portfolio_values,
            'dates': dates,
            'transaction_costs': transaction_costs_history,
            'sharpe_ratio': sharpe_ratio,
            'rebalance_dates': rebalance_dates
        }
        
    except Exception as e:
        st.error(f"Error in portfolio rebalancing: {str(e)}")
        return None

# 6. Improved Visualization Dashboard
def portfolio_dashboard(portfolio_data):
    if portfolio_data is None:
        st.error("No portfolio data available for visualization.")
        return
        
    st.title("📈 Adaptive Portfolio Optimization Dashboard")
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["Performance", "Rebalancing", "Weights", "Metrics"])
    
    with tab1:
        st.subheader("Portfolio Performance")
        
        # Portfolio value chart
        df_value = pd.DataFrame({
            "Date": portfolio_data['dates'],
            "Portfolio Value": portfolio_data['values']
        })
        
        fig1 = px.line(
            df_value, 
            x="Date", 
            y="Portfolio Value",
            title="Portfolio Value Over Time",
            labels={"Portfolio Value": "Value ($)"}
        )
        fig1.update_layout(xaxis_title="Date", yaxis_title="Portfolio Value ($)")
        st.plotly_chart(fig1, use_container_width=True)
        
        # Calculate and display returns
        if len(portfolio_data['values']) > 1:
            total_return = (portfolio_data['values'][-1] / portfolio_data['values'][0] - 1) * 100
            st.metric("Total Return", f"{total_return:.2f}%")
    
    with tab2:
        st.subheader("Rebalancing Analysis")
        
        # Transaction costs chart
        df_costs = pd.DataFrame({
            "Date": portfolio_data['dates'],
            "Transaction Cost": portfolio_data['transaction_costs']
        })
        
        fig2 = px.bar(
            df_costs, 
            x="Date", 
            y="Transaction Cost",
            title="Transaction Costs",
            labels={"Transaction Cost": "Cost ($)"}
        )
        fig2.update_layout(xaxis_title="Date", yaxis_title="Transaction Cost ($)")
        st.plotly_chart(fig2, use_container_width=True)
        
        # Rebalancing events chart
        if portfolio_data['rebalance_dates']:
            # Create a dataframe with rebalancing markers
            rebalance_markers = []
            for date in portfolio_data['dates']:
                if date in portfolio_data['rebalance_dates']:
                    rebalance_markers.append(1)  # Marker for rebalancing
                else:
                    rebalance_markers.append(0)  # No rebalancing
            
            df_rebalance = pd.DataFrame({
                "Date": portfolio_data['dates'],
                "Rebalancing Event": rebalance_markers
            })
            
            fig3 = px.scatter(
                df_rebalance[df_rebalance["Rebalancing Event"] == 1], 
                x="Date", 
                y=[portfolio_data['values'][i] for i, date in enumerate(portfolio_data['dates']) 
                   if date in portfolio_data['rebalance_dates']],
                title="Rebalancing Events",
                size=[20] * len(portfolio_data['rebalance_dates']),
                color_discrete_sequence=["red"]
            )
            fig3.update_layout(
                xaxis_title="Date", 
                yaxis_title="Portfolio Value at Rebalancing ($)",
                showlegend=False
            )
            
            # Overlay with portfolio value line
            fig3.add_scatter(
                x=portfolio_data['dates'], 
                y=portfolio_data['values'],
                mode='lines',
                line=dict(color='blue', width=1),
                name="Portfolio Value"
            )
            
            st.plotly_chart(fig3, use_container_width=True)
            
            # Rebalancing statistics
            st.metric("Number of Rebalances", len(portfolio_data['rebalance_dates']))
            total_transaction_cost = sum(portfolio_data['transaction_costs'])
            st.metric("Total Transaction Costs", f"${total_transaction_cost:.2f}")
    
    with tab3:
        st.subheader("Portfolio Weights Over Time")
        
        # Get the weights as a dataframe
        weights_df = pd.DataFrame(portfolio_data['weights']).T
        
        # Ensure column names are set
        if isinstance(weights_df.columns, pd.RangeIndex):
            weights_df.columns = [f"Asset {i+1}" for i in range(weights_df.shape[1])]
        
        # Create a long format dataframe for Plotly
        weights_long = weights_df.reset_index().melt(
            id_vars='index', 
            value_vars=weights_df.columns,
            var_name='Asset',
            value_name='Weight'
        )
        weights_long.rename(columns={'index': 'Date'}, inplace=True)
        
        # Create the area chart
        fig4 = px.area(
            weights_long, 
            x="Date", 
            y="Weight", 
            color="Asset",
            title="Portfolio Weights Over Time",
            labels={"Weight": "Allocation %"}
        )
        fig4.update_layout(xaxis_title="Date", yaxis_title="Weight", yaxis_tickformat='.0%')
        st.plotly_chart(fig4, use_container_width=True)
        
        # Display final weights
        st.subheader("Final Portfolio Allocation")
        final_weights = weights_df.iloc[-1]
        
        # Create a pie chart of final weights
        fig5 = px.pie(
            names=final_weights.index,
            values=final_weights.values,
            title="Final Portfolio Weights"
        )
        st.plotly_chart(fig5, use_container_width=True)
    
    with tab4:
        st.subheader("Performance Metrics")
        
        # Calculate metrics if we have sufficient data
        if len(portfolio_data['values']) > 30:  # At least a month of data
            portfolio_returns = pd.Series(portfolio_data['values']).pct_change().dropna()
            
            # Display Sharpe ratio
            if portfolio_data['sharpe_ratio'] is not None:
                st.metric("Sharpe Ratio", f"{portfolio_data['sharpe_ratio']:.2f}")
            
            # Calculate and display volatility
            volatility = portfolio_returns.std() * np.sqrt(252) * 100  # Annualized
            st.metric("Annualized Volatility", f"{volatility:.2f}%")
            
            # Calculate and display maximum drawdown
            cumulative_returns = (1 + portfolio_returns).cumprod()
            max_drawdown = ((cumulative_returns / cumulative_returns.cummax()) - 1).min() * 100
            st.metric("Maximum Drawdown", f"{max_drawdown:.2f}%")
            
            # Calculate and display annualized return
            days = len(portfolio_returns)
            annualized_return = ((portfolio_data['values'][-1] / portfolio_data['values'][0]) ** (252/days) - 1) * 100
            st.metric("Annualized Return", f"{annualized_return:.2f}%")
        else:
            st.info("Insufficient data for detailed performance metrics. Need at least 30 days of data.")

# 7. Improved Main Function
def main():
    st.set_page_config(
        page_title="Portfolio Optimizer",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Sidebar for user inputs
    st.sidebar.title("Portfolio Settings")
    
    # Input for tickers with validation
    ticker_input = st.sidebar.text_input(
        "Enter Ticker Symbols (comma-separated)",
        "AAPL,MSFT,GOOGL,AMZN,TSLA"
    )
    
    # Clean and validate tickers
    tickers = [ticker.strip().upper() for ticker in ticker_input.split(',') if ticker.strip()]
    
    if not tickers:
        st.sidebar.error("Please enter at least one valid ticker symbol.")
        return
    
    st.sidebar.write(f"Selected tickers: {', '.join(tickers)}")
    
    # Date range selection with validation
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            datetime.now() - timedelta(days=365),
            max_value=datetime.now() - timedelta(days=30)
        )
    
    with col2:
        end_date = st.date_input(
            "End Date",
            datetime.now(),
            min_value=start_date + timedelta(days=30),
            max_value=datetime.now()
        )
    
    if start_date >= end_date:
        st.sidebar.error("End date must be after start date.")
        return
    
    # Risk and cost parameters
    risk_aversion = st.sidebar.slider(
        "Risk Aversion",
        min_value=0.1,
        max_value=10.0,
        value=1.0,
        step=0.1,
        help="Higher values prioritize risk reduction over returns"
    )
    
    transaction_cost = st.sidebar.slider(
        "Transaction Cost (%)",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.05,
        format="%.2f"
    ) / 100  # Convert percentage to decimal
    
    weight_deviation_threshold = st.sidebar.slider(
        "Rebalancing Threshold (%)",
        min_value=1.0,
        max_value=20.0,
        value=5.0,
        step=0.5,
        help="Trigger rebalancing when any asset weight deviates by this percentage"
    ) / 100  # Convert percentage to decimal
    
    initial_capital = st.sidebar.number_input(
        "Initial Capital ($)",
        min_value=1000,
        max_value=10000000,
        value=100000,
        step=10000
    )
    
    # Run optimization button
    if st.sidebar.button("Run Portfolio Optimization", type="primary"):
        with st.spinner("Optimizing portfolio... This may take a moment."):
            # Run the portfolio optimization
            portfolio_data = trigger_rebalancing(
                tickers,
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d'),
                initial_capital=initial_capital,
                risk_aversion=risk_aversion,
                transaction_cost=transaction_cost,
                weight_deviation_threshold=weight_deviation_threshold
            )
            
            # Display the dashboard if we have data
            if portfolio_data is not None:
                portfolio_dashboard(portfolio_data)
            else:
                st.error("Portfolio optimization failed. Please check inputs and try again.")
    
    # Show information when no optimization has been run yet
    if 'portfolio_data' not in locals():
        st.title("📊 Portfolio Optimization Tool")
        st.write("""
        This tool optimizes your investment portfolio using modern portfolio theory and trigger-based rebalancing.
        
        ### How to use:
        1. Enter ticker symbols in the sidebar (e.g., AAPL, MSFT, GOOGL)
        2. Select a date range for historical data
        3. Adjust risk and cost parameters as needed
        4. Click "Run Portfolio Optimization" to see results
        
        The optimizer will create an efficient portfolio based on historical performance and rebalance when asset weights drift beyond your threshold.
        """)
        
        st.info("Enter your parameters in the sidebar and click 'Run Portfolio Optimization' to begin.")

if __name__ == "__main__":
    main()
