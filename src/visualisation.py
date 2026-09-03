import matplotlib.pyplot as plt


def plot_portfolio_vs_benchmark(data):
    """
    Plot cumulative portfolio performance
    against the benchmark.
    """

    plt.figure(figsize=(10, 6))

    plt.plot(
        data["Date"],
        data["Portfolio_Growth"],
        label="Portfolio",
    )

    plt.plot(
        data["Date"],
        data["Benchmark_Growth"],
        label="Benchmark",
    )

    plt.title("Portfolio vs Benchmark")
    plt.xlabel("Date")
    plt.ylabel("Growth of $1")

    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("data/portfolio_vs_benchmark.png")
    plt.close()

def plot_drawdown(data):
    """
    Plot portfolio drawdown over time.
    """

    plt.figure(figsize=(10, 6))

    plt.plot(
        data["Date"],
        data["Drawdown"],
    )

    plt.title("Portfolio Drawdown")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")

    plt.grid(True)

    plt.tight_layout()
    plt.savefig("data/drawdown.png")
    plt.close()

def plot_rolling_volatility(data):
    """
    Plot 60-day rolling annualised volatility.
    """

    rolling_volatility = (
        data["Portfolio_Return"]
        .rolling(60)
        .std()
        * (252 ** 0.5)
    )

    plt.figure(figsize=(10, 6))

    plt.plot(
        data["Date"],
        rolling_volatility,
    )

    plt.title("60-Day Rolling Volatility")
    plt.xlabel("Date")
    plt.ylabel("Annualised Volatility")

    plt.grid(True)

    plt.tight_layout()
    plt.savefig("data/rolling_volatility.png")
    plt.close()

def plot_rolling_sharpe(data):
    """
    Plot 60-day rolling annualised Sharpe ratio.
    """

    rolling_return = (
        data["Portfolio_Return"]
        .rolling(60)
        .mean()
    )

    rolling_volatility = (
        data["Portfolio_Return"]
        .rolling(60)
        .std()
    )

    rolling_sharpe = (
        rolling_return
        / rolling_volatility
        * (252 ** 0.5)
    )

    plt.figure(figsize=(10, 6))

    plt.plot(
        data["Date"],
        rolling_sharpe,
    )

    plt.title("60-Day Rolling Sharpe Ratio")
    plt.xlabel("Date")
    plt.ylabel("Sharpe Ratio")

    plt.grid(True)

    plt.tight_layout()
    plt.savefig("data/rolling_sharpe.png")
    plt.close()

def plot_risk_summary(risk_report, output_path):
    """
    Plot key portfolio risk measures.
    """

    metrics = [
        "Annualised Volatility",
        "Maximum Drawdown",
        "95% VaR",
        "99% VaR",
        "95% Expected Shortfall",
    ]

    values = [
        risk_report[metric]
        for metric in metrics
    ]

    plt.figure(figsize=(10, 6))

    plt.bar(metrics, values)

    plt.title("Portfolio Risk Summary")
    plt.ylabel("Risk / Return")
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()

    plt.savefig(output_path, dpi=150)

    plt.close()

def plot_risk_contributions(
    risk_contributions,
    output_path,
):
    """
    Plot portfolio risk contribution by asset.
    """

    contributions = (
        risk_contributions[
            "Component Contribution"
        ]
    )

    plt.figure(figsize=(8, 6))

    plt.bar(
        contributions.index,
        contributions.values,
    )

    plt.title("Portfolio Risk Contributions")
    plt.ylabel("Annualised Risk Contribution")

    plt.xticks(
        rotation=45,
        ha="right",
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=150,
    )

    plt.close()

def plot_rolling_volatility(
    rolling_volatility,
    output_path,
):
    """
    Plot rolling annualised portfolio volatility.
    """

    plt.figure(figsize=(10, 6))

    plt.plot(
        rolling_volatility.index,
        rolling_volatility.values,
    )

    plt.title(
        "Rolling Portfolio Volatility"
    )

    plt.ylabel("Annualised Volatility")
    plt.xlabel("Observation")

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=150,
    )

    plt.close()


def plot_rolling_tracking_error(
    rolling_tracking_error,
    output_path,
):
    """
    Plot rolling annualised tracking error.
    """

    plt.figure(figsize=(10, 6))

    plt.plot(
        rolling_tracking_error.index,
        rolling_tracking_error.values,
    )

    plt.title(
        "Rolling Tracking Error"
    )

    plt.ylabel("Annualised Tracking Error")
    plt.xlabel("Observation")

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=150,
    )

    plt.close()

def plot_cumulative_performance(
    portfolio_returns,
    benchmark_returns,
    output_path,
):
    """
    Plot cumulative portfolio and benchmark performance.
    """

    portfolio_growth = (
        1 + portfolio_returns
    ).cumprod()

    benchmark_growth = (
        1 + benchmark_returns
    ).cumprod()

    plt.figure(figsize=(10, 6))

    plt.plot(
        portfolio_growth.index,
        portfolio_growth.values,
        label="Portfolio",
    )

    plt.plot(
        benchmark_growth.index,
        benchmark_growth.values,
        label="Benchmark",
    )

    plt.title(
        "Cumulative Portfolio vs Benchmark"
    )

    plt.ylabel("Growth of $1")
    plt.xlabel("Observation")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=150,
    )

    plt.close()