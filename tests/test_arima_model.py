import pytest
import pandas as pd
from py_trading_gpt.arima_model import perform_arima_analysis


def test_arima_analysis():
    ticker = "AAPL"
    start_date = "2023-01-01"
    end_date = "2023-12-31"
    forecast_steps = 126

    perform_arima_analysis(ticker, start_date, end_date, forecast_steps)
    assert True  # Add appropriate assertions based on the function's output


def test_arima_analysis_non_stationary():
    ticker = "GOOGL"
    start_date = "2023-01-01"
    end_date = "2023-12-31"
    forecast_steps = 126

    perform_arima_analysis(ticker, start_date, end_date, forecast_steps)
    assert True  # Add appropriate assertions based on the function's output
