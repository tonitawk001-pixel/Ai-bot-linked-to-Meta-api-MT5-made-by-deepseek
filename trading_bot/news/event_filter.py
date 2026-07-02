"""Event filter — determines which news items are relevant based on active trading symbols and strategy horizon."""
from datetime import datetime, timedelta
from trading_bot.config import Config

TRADING_CURRENCIES = ["USD", "EUR", "GBP", "JPY", "XAU"]

MIN_RELEVANCE_SCORE = 0.1
MAX_AGE_HOURS = 24


def filter_events(raw_items: list, classifications: list) -> list:
    """
    Filter and merge raw news items with their classifications.

    Returns only items that are relevant to trading currencies
    and recent enough to act on.
    """
    now = datetime.now()
    filtered = []

    for item, classification in zip(raw_items, classifications):
        age = (now - item.get("published", now)).total_seconds() / 3600
        if age > MAX_AGE_HOURS:
            continue

        currencies = classification.get("currencies", {})
        has_trading_currency = any(c in TRADING_CURRENCIES for c in currencies)

        if not has_trading_currency and classification.get("impact") == "low":
            continue

        if classification.get("impact") == "high":
            item["priority"] = "high"
        elif classification.get("impact") == "medium" and has_trading_currency:
            item["priority"] = "medium"
        else:
            item["priority"] = "low"

        item["classification"] = classification
        filtered.append(item)

    filtered.sort(key=lambda x: {"high": 0, "medium": 1, "low": 2}.get(x.get("priority", "low"), 3))
    return filtered[:50]