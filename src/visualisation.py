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