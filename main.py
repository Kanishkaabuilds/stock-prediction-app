import streamlit as st
from datetime import date

import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

st.set_page_config(page_title = "Stock Prediction App", page_icon = "📈")

start = "2018-01-01"
today = date.today().strftime("%Y-%m-%d") # to get the time in str format

st.title("Stock Prediction App")
stocks = ("AAPL", "GOOG","MSFT", "GME", "AMZN", "TSLA", "NVDA") #GME - GameStop
selected_stock = st.selectbox("Select dataset for prediction", stocks)

# slider to select the no.of years
years = st.slider("Years of prediction :", 1, 5)
# no.of days
period = years * 365

@st.cache_data # while switching btw datasets no need to download dataset each time

# load stock data
def load_data(stock): #  load data for the selected stock
    data = yf.download(stock, start, today) # download data for the selected stock from start date to end date
    # returns data in panda dataframe
    data.reset_index(inplace = True) # puts date in the first column
    return data

data_load_state = st.text("Load data....")
data = load_data(selected_stock)
data_load_state.text("Loading data .... done! ")

st.subheader("Raw Data")
st.write(data.tail())

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x = data['Date'], y = data['Open'], name = 'stock_open'))
    fig.add_trace(go.Scatter(x = data['Date'], y = data['Close'], name = 'stock_close'))
    fig.layout.update(title_text = "Time Series Data", xaxis_rangeslider_visible = True)
    st.plotly_chart(fig)

plot_raw_data()

# Forecasting with fb prophet
df_train = data[['Date', 'Close']] # dataframe_train
df_train = df_train.rename(columns = {"Date": "ds", "Close": "y"})

model = Prophet()
model.fit(df_train)
# future prediction
future = model.make_future_dataframe(periods= period)
forecast = model.predict(future)

st.subheader("Forecast Data")
st.write(forecast.tail())

st.write("Forecast data")
fig1 = plot_plotly(model, forecast)
st.plotly_chart(fig1)

st.write("Forecast components")
# no need plotly graph only normal graph
fig2 = model.plot_components(forecast)
st.write(fig2)