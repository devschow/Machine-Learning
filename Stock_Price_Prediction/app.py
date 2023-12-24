import streamlit as st
from datetime import date 
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

st.title("Stock Price Prediction App")
st.write("Welcome to your prediction webapp!")

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

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

selected_stock_name = st.selectbox('Select stock for prediction', list(stocks.values()), help="Select the stock you want to predict.")
selected_stock_ticker = next((ticker for ticker, name in stocks.items() if name == selected_stock_name), "Unknown")

n_years = st.slider('Years of prediction:', 1, 4)
period = n_years * 365


@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text('Loading data...')
try:
    data = load_data(selected_stock_ticker)
    data_load_state.text('Loading data... done!')
    st.success("Data loaded successfully!")
except Exception as e:
    data_load_state.text('Loading data... error!')
    st.error(f"Error loading data: {e}")

st.subheader('Raw data')
st.write(data.tail())

# Plot raw data
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="Stock Open", line=dict(color='red')))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Stock Close", line=dict(color='blue')))
    fig.update_layout(title_text=f'Time Series Data for {selected_stock_name}', xaxis_rangeslider_visible=True, xaxis_title="Time",yaxis_title="Stock Price (USD)")
    st.plotly_chart(fig)

plot_raw_data()

# Predict forecast with Prophet.
df_train = data[['Date','Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Show and plot forecast
st.subheader('Forecast data')
st.write(forecast.tail())

st.write(f'Forecast plot for {n_years} years:')
def plot_future_data():
    fig1 = plot_plotly(m, forecast)
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="Data Values", line=dict(color='blue')))
    fig1.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name="Forecast", line=dict(color='green')))
    fig1.update_layout(showlegend=True,title_text=f'{selected_stock_name} Stock Price Forecast', xaxis_rangeslider_visible=True, xaxis_title="Time",yaxis_title="Stock Price (USD)")
    st.plotly_chart(fig1)

plot_future_data()

st.write("Forecast components")
fig2 = m.plot_components(forecast)
st.write(fig2)
