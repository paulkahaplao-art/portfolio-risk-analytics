def calculate_return(start_value, end_value):
    """Calculate the simple investment return."""
    return (end_value / start_value) - 1


if __name__ == "__main__":
    start_value = 100_000
    end_value = 108_000

    portfolio_return = calculate_return(
        start_value,
        end_value
    )

    print(f"Portfolio return: {portfolio_return:.2%}")
    