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
    """Download and clean stock data, flattening multi-level columns if needed."""
    try:
        df = yf.download(ticker, START, TODAY, group_by='ticker', progress=False)

        if df.empty:
            return pd.DataFrame()

        # Flatten multi-index columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in df.columns]

        df.reset_index(inplace=True)

        # Try to extract a usable Close column
        close_cols = [col for col in df.columns if 'Close' in col and col != 'Adj Close']
        if close_cols:
            df['Close'] = df[close_cols[0]]
        else:
            return pd.DataFrame()  # Can't proceed without Close

        return df
    except Exception as e:
        st.error(f"Data loading error: {e}")
        return pd.DataFrame()

# Load and Validate Data
data_load_state = st.text("Loading data...")
data = load_data(selected_stock_ticker)
data_load_state.text("Loading data... done!")

if data.empty:
    st.error("No data was downloaded or no usable 'Close' price found. Please try a different stock.")
    st.stop()

# Ensure required columns
if 'Date' not in data.columns or 'Close' not in data.columns:
    st.error("Missing required columns 'Date' or 'Close'.")
    st.write("Available columns:", list(data.columns))
    st.stop()

# Convert and clean
data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
data = data.dropna(subset=['Date', 'Close'])

if data.empty:
    st.error("No usable data after cleaning. Please try another stock.")
    st.stop()

st.success("Data loaded successfully!")
st.write("Downloaded data preview:")
st.write(data.head())

# Plot Raw Data
st.subheader("Raw data")
def plot_raw_data():
    fig = go.Figure()
    if 'Open' in data.columns:
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

# Prepare data for Prophet
df_train = data[['Date', 'Close']].copy()
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

if df_train.empty:
    st.error("No valid data available for training. Cannot train the model.")
    st.stop()

# Train model
st.write("Training the Prophet model...")
try:
    m = Prophet()
    m.fit(df_train)
    st.success("Model trained successfully!")
except Exception as e:
    st.error(f"Error during model training: {e}")
    st.stop()

# Make prediction
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Display results
st.subheader("Forecast data")
st.write(forecast.tail())

st.write(f'Forecast plot for {n_years} years:')
def plot_future_data():
    fig = plot_plotly(m, forecast)
    fig.update_layout(
        title_text=f'{selected_stock_name} Stock Price Forecast',
        xaxis_rangeslider_visible=True,
        xaxis_title="Time",
        yaxis_title="Stock Price (USD)"
    )
    st.plotly_chart(fig)

plot_future_data()

# Forecast components
st.write("Forecast components")
fig2 = m.plot_components(forecast)
st.write(fig2)
