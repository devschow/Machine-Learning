import streamlit as st
from datetime import date
import pandas as pd
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

# Title and Intro
st.title("Stock Price Prediction App")
st.write("Welcome to your prediction web app!")

# Constants
START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

# Stock Options
stocks = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "GOOGL": "Alphabet Inc.",
    "AMZN": "Amazon",
    "TSLA": "Tesla",
    "JPM": "JPMorgan Chase",
    "BAC": "Bank of America",
    "GOOG": "Alphabet Inc. (Class C)",
    "IBM": "International Business Machines",
    "MA": "Mastercard",
}

# User Input
selected_stock_name = st.selectbox(
    "Select stock for prediction",
    list(stocks.values()),
    help="Select the stock you want to predict."
)
selected_stock_ticker = next((ticker for ticker, name in stocks.items() if name == selected_stock_name), None)

n_years = st.slider("Years of prediction:", 1, 4)
period = n_years * 365

# Load Data Function
@st.cache_data
def load_data(ticker):
    """Download stock data and validate its structure."""
    try:
        data = yf.download(ticker, START, TODAY)
        if data.empty:
            st.error("No data was downloaded. The ticker might be incorrect or no data is available.")
            return pd.DataFrame()
        data.reset_index(inplace=True)
        return data
    except Exception as e:
        st.error(f"Error loading data for {ticker}: {e}")
        return pd.DataFrame()

# Load and Validate Data
data_load_state = st.text("Loading data...")
data = load_data(selected_stock_ticker)
data_load_state.text("Loading data... done!")

if data.empty or 'Date' not in data.columns or 'Close' not in data.columns:
    st.error("The data is missing required columns. Please check the stock symbol and try again.")
    st.stop()

# Ensure correct data types
data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
data = data.dropna(subset=['Date', 'Close'])

st.success("Data loaded successfully!")
st.write("Downloaded data preview:")
st.write(data.head())

# Plot Raw Data
st.subheader("Raw data")
def plot_raw_data():
    """Plot the raw stock data."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="Stock Open", line=dict(color='red')))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Stock Close", line=dict(color='blue')))
    fig.update_layout(
        title_text=f'Time Series Data for {selected_stock_name}',
        xaxis_rangeslider_visible=True,
        xaxis_title="Time",
        yaxis_title="Stock Price (USD)"
    )
    st.plotly_chart(fig)

plot_raw_data()

# Prepare Data for Prophet
df_train = data[['Date', 'Close']].copy()
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

# Validate Training Data
if df_train.empty:
    st.error("No valid data available for training. Cannot train the model.")
    st.stop()

# Train Prophet Model
st.write("Training the Prophet model...")
try:
    m = Prophet()
    m.fit(df_train)
    st.success("Model trained successfully!")
except Exception as e:
    st.error(f"Error during model training: {e}")
    st.stop()

# Create Future Dataframe and Predict
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Display Forecast Data
st.subheader("Forecast data")
st.write(forecast.tail())

# Plot the Forecast
st.write(f'Forecast plot for {n_years} years:')
def plot_future_data():
    """Plot the forecast data."""
    fig = plot_plotly(m, forecast)
    fig.update_layout(
        title_text=f'{selected_stock_name} Stock Price Forecast',
        xaxis_rangeslider_visible=True,
        xaxis_title="Time",
        yaxis_title="Stock Price (USD)"
    )
    st.plotly_chart(fig)

plot_future_data()

# Forecast Components
st.write("Forecast components")
fig2 = m.plot_components(forecast)
st.write(fig2)
