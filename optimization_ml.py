import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

class MLReturnPredictor:
    """
    Machine learning-based asset return prediction model.
    This class handles feature engineering, model training, and return prediction
    for portfolio optimization enhancement.
    """
    
    def __init__(self, model_type='random_forest', prediction_horizon=30, feature_window=90):
        """
        Initialize the predictor with the selected model type and parameters.
        
        Parameters:
        -----------
        model_type : str
            Type of ML model to use ('random_forest', 'gradient_boosting', or 'elastic_net')
        prediction_horizon : int
            Number of days ahead to predict returns
        feature_window : int
            Number of days of historical data to use for feature engineering
        """
        self.model_type = model_type
        self.prediction_horizon = prediction_horizon
        self.feature_window = feature_window
        self.models = {}  # Dictionary to store one model per asset
        self.scalers = {}  # Dictionary to store one scaler per asset
        self.feature_names = None  # Will be set during feature engineering
    
    def _create_features(self, prices, ticker):
        """
        Create features from price time series for a single asset.
        
        Parameters:
        -----------
        prices : pd.Series or pd.DataFrame
            Price time series for one asset
        ticker : str
            Ticker symbol of the asset
            
        Returns:
        --------
        pd.DataFrame
            Dataframe with engineered features
        """
        if isinstance(prices, pd.DataFrame):
            # Handle case when we have a dataframe with multiple price columns
            if ticker in prices.columns:
                price_series = prices[ticker]
            else:
                raise ValueError(f"Ticker {ticker} not found in price dataframe columns")
        else:
            # Assume we have a Series
            price_series = prices
        
        # Create DataFrame for features
        df = pd.DataFrame(index=price_series.index)
        
        # Price-based features
        df['price'] = price_series
        
        # Returns over different periods
        df['return_1d'] = price_series.pct_change(1)
        df['return_5d'] = price_series.pct_change(5)
        df['return_10d'] = price_series.pct_change(10)
        df['return_20d'] = price_series.pct_change(20)
        
        # Moving averages
        df['ma_5d'] = price_series.rolling(window=5).mean()
        df['ma_10d'] = price_series.rolling(window=10).mean()
        df['ma_20d'] = price_series.rolling(window=20).mean()
        df['ma_50d'] = price_series.rolling(window=50).mean()
        
        # MA ratios (momentum indicators)
        df['ma_ratio_5_20'] = df['ma_5d'] / df['ma_20d']
        df['ma_ratio_10_50'] = df['ma_10d'] / df['ma_50d']
        
        # Volatility measures
        df['volatility_10d'] = price_series.pct_change().rolling(window=10).std()
        df['volatility_20d'] = price_series.pct_change().rolling(window=20).std()
        
        # RSI (Relative Strength Index)
        delta = price_series.diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        df['rsi_14d'] = 100 - (100 / (1 + rs))
        
        # Create target variable (future return over prediction horizon)
        df['target'] = price_series.pct_change(self.prediction_horizon).shift(-self.prediction_horizon)
        
        # Drop rows with NaN values
        df = df.dropna()
        
        # Store feature names (excluding the target)
        self.feature_names = [col for col in df.columns if col != 'target']
        
        return df
    
    def _get_model(self):
        """
        Initialize the selected machine learning model.
        
        Returns:
        --------
        sklearn estimator
            Initialized model based on model_type
        """
        if self.model_type == 'random_forest':
            return RandomForestRegressor(n_estimators=100, random_state=42)
        elif self.model_type == 'gradient_boosting':
            return GradientBoostingRegressor(n_estimators=100, random_state=42)
        elif self.model_type == 'elastic_net':
            return ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def train(self, prices, tickers):
        """
        Train predictive models for each asset.
        
        Parameters:
        -----------
        prices : pd.DataFrame
            Historical price data for all assets
        tickers : list
            List of ticker symbols
            
        Returns:
        --------
        dict
            Dictionary of training metrics by ticker
        """
        training_metrics = {}
        
        # Ensure we have the right data format
        if isinstance(prices, pd.Series) and len(tickers) == 1:
            # Convert Series to DataFrame for consistent processing
            ticker = tickers[0]
            prices = pd.DataFrame({ticker: prices})
        
        for ticker in tickers:
            try:
                # Create features for this asset
                features_df = self._create_features(prices, ticker)
                
                if len(features_df) < self.feature_window:
                    st.warning(f"Insufficient data for {ticker} to train ML model. Need at least {self.feature_window} data points.")
                    continue
                
                # Split into features and target
                X = features_df[self.feature_names]
                y = features_df['target']
                
                # Create time series splits for validation
                tscv = TimeSeriesSplit(n_splits=5)
                
                # Create and fit scaler
                scaler = StandardScaler()
                self.scalers[ticker] = scaler
                
                # Create model pipeline with preprocessing
                model = Pipeline([
                    ('scaler', scaler),
                    ('regressor', self._get_model())
                ])
                
                # Train the model with time series validation
                cv_scores = []
                for train_idx, val_idx in tscv.split(X):
                    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                    
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_val)
                    
                    # Calculate validation metrics
                    mse = mean_squared_error(y_val, y_pred)
                    r2 = r2_score(y_val, y_pred)
                    cv_scores.append((mse, r2))
                
                # Train the final model on all data
                model.fit(X, y)
                self.models[ticker] = model
                
                # Store metrics
                avg_mse = np.mean([score[0] for score in cv_scores])
                avg_r2 = np.mean([score[1] for score in cv_scores])
                
                training_metrics[ticker] = {
                    'mse': avg_mse,
                    'r2': avg_r2,
                    'rmse': np.sqrt(avg_mse),
                    'feature_importance': self._get_feature_importance(model, ticker)
                }
                
            except Exception as e:
                st.error(f"Error training model for {ticker}: {str(e)}")
                training_metrics[ticker] = {'error': str(e)}
        
        return training_metrics
    
    def _get_feature_importance(self, model, ticker):
        """
        Extract feature importance from the trained model.
        
        Parameters:
        -----------
        model : sklearn model
            Trained model
        ticker : str
            Ticker symbol
            
        Returns:
        --------
        dict
            Feature importance dictionary
        """
        try:
            # Get the actual regressor from the pipeline
            regressor = model.named_steps['regressor']
            
            # Different models have different ways to get feature importance
            if hasattr(regressor, 'feature_importances_'):
                importances = regressor.feature_importances_
            elif hasattr(regressor, 'coef_'):
                importances = np.abs(regressor.coef_)
            else:
                return None
            
            # Create a dictionary of feature importances
            return dict(zip(self.feature_names, importances))
            
        except Exception as e:
            st.warning(f"Could not extract feature importance for {ticker}: {e}")
            return None
    
    def predict_returns(self, latest_prices, tickers):
        """
        Predict future returns for each asset based on the latest price data.
        
        Parameters:
        -----------
        latest_prices : pd.DataFrame
            Recent price data for feature creation
        tickers : list
            List of ticker symbols to predict
            
        Returns:
        --------
        pd.Series
            Predicted returns for each asset
        """
        predicted_returns = {}
        
        for ticker in tickers:
            if ticker not in self.models:
                st.warning(f"No trained model found for {ticker}. Using historical mean instead.")
                # Fallback to historical average return if no model is available
                historical_mean = latest_prices[ticker].pct_change().mean() * self.prediction_horizon
                predicted_returns[ticker] = historical_mean
                continue
            
            try:
                # Create features from the latest price data
                features_df = self._create_features(latest_prices, ticker)
                
                # Get the latest available feature set
                latest_features = features_df[self.feature_names].iloc[-1:]
                
                # Make prediction
                predicted_return = self.models[ticker].predict(latest_features)[0]
                predicted_returns[ticker] = predicted_return
                
            except Exception as e:
                st.error(f"Error predicting returns for {ticker}: {str(e)}")
                # Fallback to historical mean
                historical_mean = latest_prices[ticker].pct_change().mean() * self.prediction_horizon
                predicted_returns[ticker] = historical_mean
        
        return pd.Series(predicted_returns)
    
    def get_feature_importance_plot(self, ticker):
        """
        Create a feature importance visualization for a specific ticker.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol
            
        Returns:
        --------
        matplotlib figure
            Feature importance plot
        """
        if ticker not in self.models:
            return None
        
        try:
            # Get feature importances
            importances = self._get_feature_importance(self.models[ticker], ticker)
            if importances is None:
                return None
            
            # Create a dataframe for plotting
            imp_df = pd.DataFrame({
                'Feature': list(importances.keys()),
                'Importance': list(importances.values())
            }).sort_values('Importance', ascending=False)
            
            # Create the plot
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(imp_df['Feature'], imp_df['Importance'])
            ax.set_title(f'Feature Importance for {ticker}')
            ax.set_xlabel('Importance')
            plt.tight_layout()
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating feature importance plot for {ticker}: {str(e)}")
            return None


# Function to integrate ML predictions with the portfolio optimization pipeline
def get_ml_predicted_returns(tickers, start_date, end_date, prediction_horizon=30, model_type='random_forest'):
    """
    Get machine learning-based return predictions for the given tickers.
    
    Parameters:
    -----------
    tickers : list
        List of ticker symbols
    start_date : str
        Start date for historical data in 'YYYY-MM-DD' format
    end_date : str
        End date for historical data in 'YYYY-MM-DD' format
    prediction_horizon : int
        Number of days ahead to predict returns
    model_type : str
        ML model type to use
        
    Returns:
    --------
    tuple
        (predicted_returns, historical_returns, training_metrics)
    """
    try:
        # Get extended historical data (need more history for feature engineering)
        extended_start = datetime.strptime(start_date, '%Y-%m-%d') - timedelta(days=365)
        extended_start_str = extended_start.strftime('%Y-%m-%d')
        
        # Fetch data
        with st.spinner("Fetching historical data for ML model training..."):
            data = yf.download(tickers, start=extended_start_str, end=end_date)
            
            if data.empty:
                st.error("No data found for the specified tickers and date range.")
                return None, None, None
            
            # Get closing prices
            if isinstance(data.columns, pd.MultiIndex):
                prices = data['Close']
            else:
                prices = data['Close'] if 'Close' in data.columns else data
        
        # Calculate historical returns for comparison
        historical_returns = prices.pct_change().mean() * prediction_horizon
        
        # Initialize and train the ML predictor
        with st.spinner("Training machine learning models for return prediction..."):
            predictor = MLReturnPredictor(
                model_type=model_type,
                prediction_horizon=prediction_horizon,
                feature_window=90
            )
            
            # Train the models
            training_metrics = predictor.train(prices, tickers)
            
            # Log training metrics
            for ticker, metrics in training_metrics.items():
                if 'error' not in metrics:
                    st.info(f"Model for {ticker} - RMSE: {metrics['rmse']:.6f}, R²: {metrics['r2']:.4f}")
        
        # Get predictions for future returns
        with st.spinner("Generating return predictions..."):
            predicted_returns = predictor.predict_returns(prices, tickers)
            
            # Display comparison
            comparison = pd.DataFrame({
                'ML Predicted Returns': predicted_returns,
                'Historical Average Returns': historical_returns
            })
            
            st.subheader("Return Predictions")
            st.dataframe(comparison)
            
            # Show feature importance for a selected ticker
            if len(tickers) > 0:
                selected_ticker = st.selectbox("Select ticker for feature importance", tickers)
                fig = predictor.get_feature_importance_plot(selected_ticker)
                if fig:
                    st.pyplot(fig)
        
        return predicted_returns, historical_returns, training_metrics, predictor
        
    except Exception as e:
        st.error(f"Error in ML return prediction: {str(e)}")
        return None, None, None, None


# Modified portfolio optimization function to use ML predictions
def optimize_portfolio_with_ml(mu, sigma, prev_weights=None, risk_aversion=0.5, 
                              short_selling=False, transaction_cost=0.001,
                              ml_returns=None, confidence=0.5):
    """
    Portfolio optimization with ML-predicted returns integration.
    
    Parameters:
    -----------
    mu : numpy array
        Historical expected returns
    sigma : numpy array
        Covariance matrix
    prev_weights : numpy array, optional
        Previous portfolio weights
    risk_aversion : float
        Risk aversion parameter
    short_selling : bool
        Whether to allow short selling
    transaction_cost : float
        Transaction cost as a fraction
    ml_returns : pandas Series, optional
        ML-predicted returns for each asset
    confidence : float
        Confidence in ML predictions (0-1)
        
    Returns:
    --------
    numpy array
        Optimal portfolio weights
    """
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
        
        # Blend historical and ML-predicted returns if available
        if ml_returns is not None and confidence > 0:
            # Convert to numpy array if needed
            if isinstance(ml_returns, pd.Series):
                ml_returns = ml_returns.values
                
            if len(ml_returns) == n:
                # Blend returns based on confidence parameter
                blended_mu = (1 - confidence) * mu + confidence * ml_returns
                st.info(f"Using ML predictions with {confidence*100:.0f}% confidence")
            else:
                st.warning("ML predictions dimension mismatch. Using historical returns only.")
                blended_mu = mu
        else:
            blended_mu = mu
            
        # Define portfolio optimization problem
        import cvxpy as cp
        w = cp.Variable(n)
        
        # Define the objective function with transaction costs if applicable
        objective = cp.Maximize(blended_mu.T @ w - risk_aversion * cp.quad_form(w, sigma))
        
        if prev_weights is not None:
            transaction_penalty = cp.norm(w - prev_weights, 1) * transaction_cost
            objective = cp.Maximize(blended_mu.T @ w - risk_aversion * cp.quad_form(w, sigma) - transaction_penalty)
        
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


# Update the main function to include ML options
def main_with_ml():
    st.set_page_config(
        page_title="ML-Enhanced Portfolio Optimizer",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Create tabs for different application sections
    tab1, tab2 = st.tabs(["Portfolio Optimization", "ML Model Analysis"])
    
    with tab1:
        st.title("📈 ML-Enhanced Portfolio Optimization")
        
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
        value=(datetime.datetime.now() - datetime.timedelta(days=365*2)),  # Use datetime.datetime
        max_value=datetime.datetime.now() - datetime.timedelta(days=30)
    )

with col2:
    end_date = st.date_input(
        "End Date",
        value=datetime.datetime.now(),
        min_value=start_date + datetime.timedelta(days=30),
        max_value=datetime.datetime.now()
    )
        if start_date >= end_date:
            st.sidebar.error("End date must be after start date.")
            return
        
        # ML settings
        st.sidebar.subheader("Machine Learning Settings")
        use_ml = st.sidebar.checkbox("Use ML for Return Prediction", value=True)
        
        ml_model_type = st.sidebar.selectbox(
            "ML Model Type",
            ["random_forest", "gradient_boosting", "elastic_net"],
            index=0,
            disabled=not use_ml
        )
        
        prediction_horizon = st.sidebar.slider(
            "Prediction Horizon (Days)",
            min_value=5,
            max_value=60,
            value=30,
            step=5,
            disabled=not use_ml
        )
        
        ml_confidence = st.sidebar.slider(
            "ML Prediction Confidence",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
            disabled=not use_ml,
            help="Weight given to ML predictions vs. historical returns"
        )
        
        # Risk and cost parameters
        st.sidebar.subheader("Portfolio Parameters")
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
            # First fetch stock data
            from datetime import datetime
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')
            
            with st.spinner("Fetching stock data..."):
                prices, returns = get_stock_data(tickers, start_str, end_str)
                
                if prices is None or returns is None:
                    st.error("Failed to retrieve stock data. Please check your inputs.")
                    return
            
            # Calculate basic metrics
            with st.spinner("Calculating portfolio metrics..."):
                mu, sigma = calculate_metrics(returns)
                
                if mu is None or sigma is None:
                    st.error("Failed to calculate portfolio metrics.")
                    return
            
            # ML predictions if enabled
            ml_returns = None
            predictor = None
            if use_ml:
                ml_results = get_ml_predicted_returns(
                    tickers, 
                    start_str, 
                    end_str,
                    prediction_horizon=prediction_horizon,
                    model_type=ml_model_type
                )
                
                if ml_results is not None and len(ml_results) >= 3:
                    ml_returns, historical_returns, training_metrics, predictor = ml_results
            
            # Run the portfolio optimization with ML integration
            with st.spinner("Optimizing portfolio..."):
                optimal_weights = optimize_portfolio_with_ml(
                    mu, 
                    sigma, 
                    None,  # No previous weights for initial allocation
                    risk_aversion=risk_aversion,
                    transaction_cost=transaction_cost,
                    ml_returns=ml_returns,
                    confidence=ml_confidence if use_ml else 0
                )
                
                if optimal_weights is None:
                    st.error("Portfolio optimization failed.")
                    return
                
                # Display the optimal weights
                weights_df = pd.DataFrame({
                    'Asset': tickers,
                    'Allocation': optimal_weights,
                    'Allocation (%)': [f"{w*100:.2f}%" for w in optimal_weights]
                })
                
                st.subheader("Optimal Portfolio Allocation")
                st.dataframe(weights_df)
                
                # Create a pie chart of the allocations
                import plotly.express as px
                fig = px.pie(
                    weights_df, 
                    values='Allocation', 
                    names='Asset',
                    title="Portfolio Allocation"
                )
                st.plotly_chart(fig)
                
                # Calculate expected portfolio metrics
                expected_return = np.sum(optimal_weights * mu) * 252  # Annualized
                portfolio_vol = np.sqrt(np.dot(optimal_weights.T, np.dot(sigma, optimal_weights))) * np.sqrt(252)  # Annualized
                sharpe_ratio = expected_return / portfolio_vol if portfolio_vol > 0 else 0
                
                # Display metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Expected Annual Return", f"{expected_return*100:.2f}%")
                with col2:
                    st.metric("Expected Annual Volatility", f"{portfolio_vol*100:.2f}%")
                with col3:
                    st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
                
                # If ML was used, show the comparison of expected returns
                if use_ml and ml_returns is not None:
                    st.subheader("Return Prediction Comparison")
                    
                    # Calculate expected portfolio return with ML predictions
                    ml_portfolio_return = np.sum(optimal_weights * ml_returns) * (252/prediction_horizon)  # Annualized
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Historical-Based Annual Return", f"{expected_return*100:.2f}%")
                    with col2:
                        st.metric("ML-Predicted Annual Return", f"{ml_portfolio_return*100:.2f}%")
                    
                    # Explanation of the blending
                    st.info(f"""
                        The portfolio optimization used a blend of:
                        - {(1-ml_confidence)*100:.0f}% weight on historical returns
                        - {ml_confidence*100:.0f}% weight on ML-predicted returns
                    """)
    
    # ML Model Analysis Tab
    with tab2:
        st.title("🧠 ML Model Analysis")
        
        if 'predictor' in locals() and predictor is not None:
            # Show feature importance for all assets
            st.subheader("Feature Importance Analysis")
            
            selected_ticker = st.selectbox(
                "Select Asset for Detailed Analysis", 
                tickers,
                key="ml_analysis_ticker"
            )
            
            # Show feature importance plot
            fig = predictor.get_feature_importance_plot(selected_ticker)
            if fig:
                st.pyplot(fig)
            
            # Show model performance metrics
            if training_metrics and selected_ticker in training_metrics:
                metrics = training_metrics[selected_ticker]
                if 'error' not in metrics:
                    st.subheader("Model Performance Metrics")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("RMSE", f"{metrics['rmse']:.6f}")
                    with col2:
                        st.metric("R² Score", f"{metrics['r2']:.4f}")
                    with col3:
                        st.metric("MSE", f"{metrics['mse']:.6f}")
                    
                    # Explanation of metrics
                    st.info("""
                        **Metrics Explained:**
                        - **RMSE**: Root Mean Squared Error measures the standard deviation of prediction errors
                        - **R² Score**: Coefficient of determination, higher is better (1.0 is perfect prediction)
                        - **MSE**: Mean Squared Error, the average of squared differences between predicted and actual values
                    """)
            
            # Show predicted vs historical returns comparison
            if ml_returns is not None and historical_returns is not None:
                st.subheader("Return Predictions Comparison")
                
                comparison = pd.DataFrame({
                    'ML Predicted Returns': ml_returns,
                    'Historical Average Returns': historical_returns,
                    'Difference': ml_returns - historical_returns
                })
                
                # Format as percentages
                formatted_comparison = comparison.copy()
                for col in formatted_comparison.columns:
                    formatted_comparison[col] = formatted_comparison[col].apply(lambda x: f"{x*100:.2f}%")
                
                st.dataframe(formatted_comparison)
                
                # Visualize the comparison
                fig = px.bar(
                    comparison.reset_index(),
                    x='index',
                    y=['ML Predicted Returns', 'Historical Average Returns'],
                    barmode='group',
                    labels={'index': 'Asset', 'value': 'Expected Return'},
                    title="ML vs Historical Return Predictions"
                )
                st.plotly_chart(fig)
        else:
            st.info("Run the portfolio optimization with ML enabled to see model analysis.")
            
            # Explanation of the ML approach
            st.markdown("""
            ### Machine Learning Approach for Return Prediction
            
            The ML-enhanced portfolio optimization uses advanced machine learning techniques to predict future asset returns:
            
            **Feature Engineering:**
            - Price trends and momentum indicators
            - Moving averages over multiple time periods
            - Volatility measures
            - Technical indicators like RSI
            
            **Model Options:**
            - **Random Forest**: Ensemble method using multiple decision trees
            - **Gradient Boosting**: Sequential ensemble method focusing on errors of previous models
            - **Elastic Net**: Linear regression with L1 and L2 regularization
            
            **Integration with Portfolio Optimization:**
            The ML predictions are blended with historical return estimates based on a confidence parameter, allowing you to control how much weight is given to the predictive models.
            
            **Benefits:**
            - More forward-looking than traditional mean-variance optimization
            - Adaptive to changing market conditions
            - Can capture complex non-linear patterns in asset returns
            """)

if __name__ == "__main__":
    main_with_ml()
