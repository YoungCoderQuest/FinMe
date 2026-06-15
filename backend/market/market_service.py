import yfinance as yf


def get_quote(symbol: str):

    ticker = yf.Ticker(symbol)

    history = ticker.history(period="2d")

    if history.empty:
        raise Exception(f"No data returned for {symbol}")

    latest = history.iloc[-1]

    previous_close = None

    if len(history) > 1:
        previous_close = history.iloc[-2]["Close"]

    price = float(latest["Close"])

    if previous_close:
        change = price - float(previous_close)

        change_pct = (
            (change / float(previous_close)) * 100
        )
    else:
        change = 0
        change_pct = 0

    return {
        "symbol": symbol,
        "price": round(price, 2),
        "change": round(change, 2),
        "change_pct": round(change_pct, 2),
        "day_high": round(float(latest["High"]), 2),
        "day_low": round(float(latest["Low"]), 2),
        "open": round(float(latest["Open"]), 2),
        "volume": int(latest["Volume"]),
    }