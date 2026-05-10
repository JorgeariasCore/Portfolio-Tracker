import yfinance as yf
import pandas as pd
import numpy as np


# =========================
# CONFIGURATION
# =========================
PORTFOLIO = {
    "AAPL": 10,
    "MSFT": 8,
    "GOOGL": 6,
    "SPY": 12
}

START_DATE = "2023-01-01"
END_DATE = "2026-01-01"


# =========================
# DOWNLOAD DATA
# =========================
def download_prices(tickers, start, end):
    data = yf.download(tickers, start=start, end=end, auto_adjust=True)

    if "Close" in data.columns:
        prices = data["Close"]
    else:
        prices = data

    prices = prices.dropna(how="all")
    return prices


# =========================
# PORTFOLIO CALCULATIONS
# =========================
def calculate_position_values(latest_prices, shares_dict):
    positions = {}

    for ticker, shares in shares_dict.items():
        positions[ticker] = latest_prices[ticker] * shares #multiple the prices with shares

    return pd.Series(positions, name="Position Value") #return the serie with name


def calculate_weights(position_values):
    total_value = position_values.sum() #summarize all the position
    weights = position_values / total_value #Price's position divide in total value, should be less that 1
    return weights


def calculate_daily_returns(prices):
    return prices.pct_change().dropna() #calculate the porcentage change between consecutive value


def calculate_portfolio_returns(daily_returns, weights):
    # Align weights to the columns order
    weights_array = np.array([weights[ticker] for ticker in daily_returns.columns])#organize the weights with daily returns
    portfolio_daily_returns = daily_returns.dot(weights_array) #computes the dot product between two arrays
    return portfolio_daily_returns #portfolio performance 


def calculate_cumulative_return(portfolio_returns):
    return (1 + portfolio_returns).cumprod() - 1 #cumulative product It multiplies values one after another down the column.


def calculate_annualized_volatility(portfolio_returns):
    return portfolio_returns.std() * np.sqrt(252)# Annualized Volatility


def calculate_correlation_matrix(daily_returns):
    return daily_returns.corr() #measure correlation

# =========================
# SUMMARY TABLE
# =========================
def build_portfolio_summary(latest_prices, shares_dict, position_values, weights):
    summary = pd.DataFrame(index=latest_prices.index)

    summary["Shares"] = pd.Series(shares_dict)
    summary["Latest Price"] = latest_prices
    summary["Position Value"] = position_values
    summary["Portfolio Weight"] = weights

    return summary.sort_values(by="Position Value", ascending=False)


# =========================
# MAIN
# =========================
def main():
    tickers = list(PORTFOLIO.keys()) #made a diccionary
    print("Downloading portfolio data...")
    prices = download_prices(tickers, START_DATE, END_DATE)

    print("\nLatest prices:")
    print(prices.tail()) # show the las 5 prices

    latest_prices = prices.iloc[-1] # pickup the last price
    daily_returns = calculate_daily_returns(prices)
    position_values = calculate_position_values(latest_prices, PORTFOLIO) # send the data to calculate portafolio 
    weights = calculate_weights(position_values) #send the data as a serie

    portfolio_returns = calculate_portfolio_returns(daily_returns, weights)
    cumulative_return = calculate_cumulative_return(portfolio_returns)
    annualized_volatility = calculate_annualized_volatility(portfolio_returns)
    correlation_matrix = calculate_correlation_matrix(daily_returns)

    summary = build_portfolio_summary(latest_prices, PORTFOLIO, position_values, weights)

    total_portfolio_value = position_values.sum() #summarize all value for a total portafolio

    print("\n================ PORTFOLIO SUMMARY ================\n")
    print(summary)

    print(f"\nTotal Portfolio Value: ${total_portfolio_value:,.2f}")

    print("\n================ PORTFOLIO PERFORMANCE ================\n")
    print(f"Latest Portfolio Daily Return: {portfolio_returns.iloc[-1]:.2%}")
    print(f"Total Cumulative Return: {cumulative_return.iloc[-1]:.2%}")
    print(f"Annualized Volatility: {annualized_volatility:.2%}")

    print("\n================ CORRELATION MATRIX ================\n")
    print(correlation_matrix)

    print("\n================ LAST 5 PORTFOLIO RETURNS ================\n")
    print(portfolio_returns.tail())


if __name__ == "__main__":
    main()