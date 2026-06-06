from __future__ import annotations
from datetime import date, timedelta
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import yfinance as yf
from curl_cffi import requests

# sidebar
BURSA_STOCKS: dict[str, str] = {
    "1155.KL": "Malayan Banking Berhad",
    "1023.KL": "CIMB Group Holdings Berhad",
    "5347.KL": "Tenaga Nasional Berhad",
    "6947.KL": "CelcomDigi Berhad",
    "8869.KL": "Press Metal Aluminium Holdings Berhad",
    "1295.KL": "Public Bank Berhad",
    "5819.KL": "Hong Leong Bank Berhad",
    "6012.KL": "Maxis Berhad",
    "2445.KL": "Kuala Lumpur Kepong Berhad",
    "1961.KL": "IOI Corporation Berhad",
    "4707.KL": "Nestle Malaysia Berhad",
    "7113.KL": "Top Glove Corporation Berhad",
    "3182.KL": "Genting Berhad",
    "4197.KL": "Sime Darby Berhad",
    "5285.KL": "Sime Darby Plantation Berhad",
}

DEFAULT_INVESTMENT_AMOUNT = 1000.00

PERIOD_OPTIONS: dict[str, int] = {
    "1 month": 30,
    "3 months": 90,
    "6 months": 180,
    "1 year": 365,
}

def get_analysis_period(period_label: str) -> tuple[date, date]:
    """Convert the selected period into a Yahoo Finance date range."""
    end_date = date.today()
    start_date = end_date - timedelta(days=PERIOD_OPTIONS[period_label])
    return start_date, end_date

@st.cache_data(show_spinner=False)
def retrieve_stock_data(tickers: tuple[str, ...], start_date: date, end_date: date) -> pd.DataFrame:
    """Retrieve historical closing prices from Yahoo Finance using yfinance."""
    yahoo_session = requests.Session(impersonate="chrome")

    yahoo_session.verify = False

    try:
        data = yf.download(
            list(tickers),
            start=start_date,
            end=end_date + timedelta(days=1),
            progress=False,
            auto_adjust=False,
            group_by="column",
            session=yahoo_session,
        )
    except Exception as exc:
        raise RuntimeError(f"Unable to retrieve stock data from Yahoo Finance: {exc}") from exc

    if data.empty:
        return pd.DataFrame()

    # returns multiindez columns
    if isinstance(data.columns, pd.MultiIndex):
        close_prices = data["Close"].copy()
    else:
        close_prices = data[["Close"]].rename(columns={"Close": tickers[0]})

    close_prices.index = pd.to_datetime(close_prices.index)
    close_prices = close_prices.dropna(how="all")
    return close_prices

def classify_performance(return_percentage: float) -> str:
    """Classify stock performance using the assignment rules."""
    if return_percentage < 0:
        return "Negative Return"
    if return_percentage <= 2:
        return "Moderate Return"
    return "High Return"

def calculate_stock_analysis(close_prices: pd.DataFrame, investment_amount: float) -> pd.DataFrame:
    """Calculate daily return, estimated return, and return percentage for each stock."""
    rows: list[dict[str, float | str]] = []

    for ticker in close_prices.columns:
        stock_series = close_prices[ticker].dropna()
        if len(stock_series) < 2:
            continue

        yesterday_close = float(stock_series.iloc[-2])
        today_close = float(stock_series.iloc[-1])
        daily_return = today_close - yesterday_close

        shares_purchasable = investment_amount / yesterday_close
        estimated_total_return = daily_return * shares_purchasable
        return_percentage = (estimated_total_return / investment_amount) * 100

        rows.append(
            {
                "Ticker": ticker,
                "Yesterday Closing Price": yesterday_close,
                "Today Closing Price": today_close,
                "Daily Return": daily_return,
                "Number of Shares Purchasable": shares_purchasable,
                "Estimated Total Return": estimated_total_return,
                "Return Percentage": return_percentage,
                "Performance Category": classify_performance(return_percentage),
            }
        )

    return pd.DataFrame(rows)

def create_portfolio_summary(stock_analysis: pd.DataFrame) -> pd.DataFrame:
    """Use pandas column selection/slicing to create the required summary table."""
    return stock_analysis.loc[
        :,
        [
            "Ticker",
            "Yesterday Closing Price",
            "Today Closing Price",
            "Estimated Total Return",
            "Return Percentage",
        ],
    ]

def create_performance_classification(stock_analysis: pd.DataFrame) -> pd.DataFrame:
    """Return the required performance classification display columns."""
    return stock_analysis.loc[:, ["Ticker", "Return Percentage", "Performance Category"]]

def create_groupby_summary(stock_analysis: pd.DataFrame) -> pd.DataFrame:
    """Use pandas groupby() to calculate average estimated return by category."""
    return (
        stock_analysis.groupby("Performance Category", as_index=False)["Estimated Total Return"]
        .mean()
        .rename(columns={"Estimated Total Return": "Average Estimated Total Return"})
    )

def generate_closing_price_chart(close_prices: pd.DataFrame) -> plt.Figure:
    """Create Chart 1: Closing Price Trend."""
    fig, ax = plt.subplots(figsize=(11, 5.5))

    for ticker in close_prices.columns:
        ax.plot(close_prices.index, close_prices[ticker], linewidth=2, marker="o", label=ticker)

    ax.set_title("Closing Price Trend")
    ax.set_xlabel("Date")
    ax.set_ylabel("Closing Price")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig

def generate_portfolio_performance_chart(stock_analysis: pd.DataFrame) -> plt.Figure:
    """Create Chart 2: Portfolio Performance Comparison."""
    sorted_df = stock_analysis.sort_values("Return Percentage", ascending=False)
    colors = ["#15803d" if value >= 0 else "#b91c1c" for value in sorted_df["Return Percentage"]]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(sorted_df["Ticker"], sorted_df["Return Percentage"], color=colors)
    ax.axhline(0, color="#111827", linewidth=1)
    ax.set_title("Portfolio Performance Comparison")
    ax.set_xlabel("Stock Ticker")
    ax.set_ylabel("Return Percentage")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig

def calculate_portfolio_insights(stock_analysis: pd.DataFrame) -> dict[str, float | str]:
    """Calculate automatic portfolio insight values for the dashboard."""
    best_row = stock_analysis.loc[stock_analysis["Return Percentage"].idxmax()]
    worst_row = stock_analysis.loc[stock_analysis["Return Percentage"].idxmin()]

    return {
        "Best Performing Stock": str(best_row["Ticker"]),
        "Worst Performing Stock": str(worst_row["Ticker"]),
        "Highest Return Percentage": float(best_row["Return Percentage"]),
        "Lowest Return Percentage": float(worst_row["Return Percentage"]),
        "Average Portfolio Return": float(stock_analysis["Return Percentage"].mean()),
        "Total Portfolio Profit/Loss": float(stock_analysis["Estimated Total Return"].sum()),
    }

def display_formatted_dataframe(df: pd.DataFrame) -> None:
    """Apply consistent formatting before displaying tables."""
    format_rules = {
        "Yesterday Closing Price": "RM {:.2f}",
        "Today Closing Price": "RM {:.2f}",
        "Daily Return": "RM {:.2f}",
        "Number of Shares Purchasable": "{:.2f}",
        "Estimated Total Return": "RM {:.2f}",
        "Average Estimated Total Return": "RM {:.2f}",
        "Return Percentage": "{:.2f}%",
    }
    active_rules = {column: rule for column, rule in format_rules.items() if column in df.columns}
    st.dataframe(df.style.format(active_rules), use_container_width=True)

def main() -> None:
    st.set_page_config(page_title="Bursa Malaysia Stock Analysis Dashboard", layout="wide")
    st.title("Bursa Malaysia Stock Analysis Dashboard")

    # Sidebar controls update immediately, while the button controls when analysis runs.
    st.sidebar.header("Inputs")
    stock_labels = {f"{ticker} - {name}": ticker for ticker, name in BURSA_STOCKS.items()}

    selected_labels = st.sidebar.multiselect(
        "Select exactly 5 Bursa Malaysia stocks",
        options=list(stock_labels.keys()),
        default=[],
        placeholder="Choose 5 stocks",
    )
    st.sidebar.caption(f"Selected: {len(selected_labels)} / 5")
    investment_amount = st.sidebar.number_input(
        "Investment Amount (RM)",
        min_value=1.00,
        value=DEFAULT_INVESTMENT_AMOUNT,
        step=1.00,
        format="%.2f",
    )
    period_label = st.sidebar.selectbox("Analysis Period", options=list(PERIOD_OPTIONS.keys()), index=0)
    analyze_clicked = st.sidebar.button(
        "Analyze Selected Stocks",
        disabled=len(selected_labels) != 5,
        type="primary",
    )

    if analyze_clicked:
        selected_tickers = tuple(stock_labels[label] for label in selected_labels)

        if len(selected_tickers) != 5:
            st.session_state.pop("analysis_request", None)
            st.warning("Please select exactly 5 Bursa Malaysia stocks, then click Analyze Selected Stocks.")
            st.stop()

        st.session_state["analysis_request"] = {
            "selected_tickers": selected_tickers,
            "investment_amount": investment_amount,
            "period_label": period_label,
        }

    if "analysis_request" not in st.session_state:
        st.info("Use the sidebar controls to choose 5 Bursa Malaysia stocks, then click Analyze Selected Stocks.")
        st.stop()

    selected_tickers = st.session_state["analysis_request"]["selected_tickers"]
    investment_amount = st.session_state["analysis_request"]["investment_amount"]
    period_label = st.session_state["analysis_request"]["period_label"]

    st.caption(
        "Current analysis: "
        f"{', '.join(selected_tickers)} | Investment Amount: RM {investment_amount:,.2f} | Period: {period_label}"
    )

    start_date, end_date = get_analysis_period(period_label)

    try:
        with st.spinner("Retrieving historical stock data from Yahoo Finance..."):
            close_prices = retrieve_stock_data(selected_tickers, start_date, end_date)
    except RuntimeError as exc:
        st.error(str(exc))
        st.stop()

    if close_prices.empty or len(close_prices) < 2:
        st.error("Market data is unavailable for the selected stocks and period. Please try another selection.")
        st.stop()

    stock_analysis = calculate_stock_analysis(close_prices, investment_amount)

    if len(stock_analysis) != 5:
        missing_count = 5 - len(stock_analysis)
        st.error(
            f"{missing_count} selected stock(s) did not return enough closing prices. "
            "Please choose different Bursa Malaysia stocks."
        )
        st.stop()

    insights = calculate_portfolio_insights(stock_analysis)

    metric_1, metric_2, metric_3, metric_4 = st.columns(4)
    metric_1.metric("Best Performing Stock", str(insights["Best Performing Stock"]))
    metric_2.metric("Worst Performing Stock", str(insights["Worst Performing Stock"]))
    metric_3.metric("Average Portfolio Return", f"{insights['Average Portfolio Return']:.2f}%")
    metric_4.metric("Total Portfolio Profit/Loss", f"RM {insights['Total Portfolio Profit/Loss']:,.2f}")

    st.header("Stock Analysis")
    display_formatted_dataframe(stock_analysis)

    csv_data = stock_analysis.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download CSV",
        data=csv_data,
        file_name="bursa_stock_analysis.csv",
        mime="text/csv",
    )

    st.header("Portfolio Summary")
    portfolio_summary = create_portfolio_summary(stock_analysis)
    display_formatted_dataframe(portfolio_summary)

    st.header("Performance Classification")
    performance_classification = create_performance_classification(stock_analysis)
    display_formatted_dataframe(performance_classification)

    st.subheader("Average Estimated Total Return by Performance Category")
    groupby_summary = create_groupby_summary(stock_analysis)
    display_formatted_dataframe(groupby_summary)

    st.header("Charts")
    chart_1, chart_2 = st.columns(2)
    chart_1.pyplot(generate_closing_price_chart(close_prices))
    chart_2.pyplot(generate_portfolio_performance_chart(stock_analysis))

    st.header("Portfolio Insights")
    insight_a, insight_b = st.columns(2)
    insight_a.write(f"**Best Performing Stock:** {insights['Best Performing Stock']}")
    insight_a.write(f"**Highest Return Percentage:** {insights['Highest Return Percentage']:.2f}%")
    insight_a.write(f"**Average Portfolio Return:** {insights['Average Portfolio Return']:.2f}%")
    insight_b.write(f"**Worst Performing Stock:** {insights['Worst Performing Stock']}")
    insight_b.write(f"**Lowest Return Percentage:** {insights['Lowest Return Percentage']:.2f}%")
    insight_b.write(f"**Total Portfolio Profit/Loss:** RM {insights['Total Portfolio Profit/Loss']:,.2f}")

if __name__ == "__main__":
    main()
