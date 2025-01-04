import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
import yfinance as yf



def perform_arima_analysis(ticker: str, start_date: str, end_date: str, forecast_steps: int):
    try:
        # Step 1: Data Collection
        data = yf.download(ticker, start=start_date, end=end_date)
        data = data["Close"]

        # Step 2: Data Preparation
        # Check for stationarity
        result = adfuller(data)
        print(f"ADF Statistic: {result[0]}")
        print(f"p-value: {result[1]}")
    data_diff = data.diff().dropna() if result[1] > 0.05 else data
        # Step 3: Model Identification
        # Plot ACF and PACF
        fig, axes = plt.subplots(1, 2, figsize=(16, 4))
        plot_acf(data_diff, ax=axes[0])
        plot_pacf(data_diff, ax=axes[1])
        plt.show()

    plt.close()
        # Determine ARIMA parameters programmatically
        p = len([i for i in plot_pacf(data_diff, lags=20).values if i > 0.2])
        q = len([i for i in plot_acf(data_diff, lags=20).values if i > 0.2])
        d = 1 if result[1] > 0.05 else 0

        # Step 4: Model Estimation
        # Fit ARIMA model (p, d, q) - determined from ACF and PACF plots
        model = ARIMA(data, order=(p, d, q))
        model_fit = model.fit()
        print(model_fit.summary())

        # Step 5: Forecasting
        # Forecasting the next forecast_steps days
        forecast = model_fit.get_forecast(steps=forecast_steps)
        forecast_index = pd.date_range(start=data.index[-1], periods=forecast_steps, freq="B")
        forecast_series = pd.Series(forecast.predicted_mean, index=forecast_index)
        conf_int = forecast.conf_int()

        # Step 6: Visualization
        plt.figure(figsize=(12, 6))
        plt.plot(data, label="Historical")
        plt.plot(forecast_series, label="Forecast", color="red")
        plt.fill_between(forecast_series.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color="pink", alpha=0.3)
        plt.title(f"{ticker} Price Forecast")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.show()

        # Explicit plot cleanup to prevent memory leaks
        plt.close("all")

    except Exception as e:
        print(f"An error occurred: {e}")
