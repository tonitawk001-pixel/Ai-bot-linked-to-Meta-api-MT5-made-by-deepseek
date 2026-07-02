"""
News sentiment analysis engine.

Classifies news items by:
- Currency impacted
- Sentiment direction (bullish/bearish/neutral)
- Impact level (low/medium/high)
- Confidence score (0-1)

Uses keyword-based classification with configurable dictionaries.
Can optionally use DeepSeek for enhanced analysis when available.
"""

import re
from datetime import datetime, timedelta
from collections import defaultdict

from trading_bot.utils.logger import logger


# Bullish and bearish keyword dictionaries per currency
SENTIMENT_DICT = {
    "bullish": {
        "USD": ["fed hawkish", "rate hike", "tightening", "strong economy", "gdp growth",
                "jobs report beat", "cpi above", "retail sales beat", "dollar strength",
                "unemployment low", "manufacturing expansion"],
        "EUR": ["ecb hawkish", "eurozone growth", "german gdp", "euro strength",
                "inflation rising", "rate hike expectations"],
        "GBP": ["boe hawkish", "uk growth", "pound strength", "inflation rising",
                "rate hike", "employment strong"],
        "JPY": ["boj hawkish", "yen strength", "inflation rising", "japan growth",
                "rate hike", "yield curve control removal"],
        "XAU": ["gold demand", "safe haven", "gold rally", "central bank buying",
                "inflation hedge", "gold etf inflows"],
    },
    "bearish": {
        "USD": ["fed dovish", "rate cut", "quantitative easing", "recession fears",
                "gdp miss", "jobs report miss", "cpi below", "retail sales miss",
                "dollar weakness", "unemployment rising"],
        "EUR": ["ecb dovish", "recession fears", "german slowdown", "euro weakness",
                "deflation risk", "political uncertainty"],
        "GBP": ["boe dovish", "uk recession", "pound weakness", "brexit uncertainty",
                "inflation falling", "economic contraction"],
        "JPY": ["boj dovish", "yen weakness", "deflation", "japan recession",
                "negative rates", "economic stagnation"],
        "XAU": ["gold selloff", "dollar strength", "rate hike pressure", "gold demand falls",
                "safe haven flows reverse", "gold etf outflows"],
    },
}

# Keywords that trigger auto-high-impact classification
HIGH_IMPACT_TRIGGERS = [
    "fed rate", "fomc", "non-farm", "nfp", "cpi", "ppi", "gdp",
    "interest rate decision", "central bank", "inflation",
    "recession", "employment", "jobs report", "retail sales",
    "trade war", "tariff", "sanctions", "default",
]


class SentimentEngine:
    """
    Classifies news items by sentiment, currency impact, and severity.
    """

    def __init__(self):
        self._bullish_dict = SENTIMENT_DICT["bullish"]
        self._bearish_dict = SENTIMENT_DICT["bearish"]

    def classify(self, item: dict) -> dict:
        """
        Classify a single news item.

        Args:
            item: News item dict with title, description.

        Returns:
            dict with classification results.
        """
        text = (item.get("title", "") + " " + item.get("description", "")).lower()

        # Determine impacted currencies
        impacted = item.get("matched_currencies", [])
        if not impacted:
            for currency in ["USD", "EUR", "GBP", "JPY", "XAU"]:
                if currency.lower() in text or _currency_aliases(currency) in text:
                    impacted.append(currency)

        # Determine sentiment per currency
        currency_sentiment = {}
        overall_sentiment_score = 0.0

        for currency in impacted:
            bullish_score = sum(1 for kw in self._bullish_dict.get(currency, [])
                               if kw in text)
            bearish_score = sum(1 for kw in self._bearish_dict.get(currency, [])
                               if kw in text)

            if bullish_score > bearish_score:
                sentiment = "bullish"
                score = min(1.0, bullish_score * 0.3)
            elif bearish_score > bullish_score:
                sentiment = "bearish"
                score = min(1.0, bearish_score * 0.3)
            else:
                sentiment = "neutral"
                score = 0.5

            currency_sentiment[currency] = {
                "sentiment": sentiment,
                "confidence": round(score, 2),
                "bullish_signals": bullish_score,
                "bearish_signals": bearish_score,
            }

            if sentiment == "bullish":
                overall_sentiment_score += score
            elif sentiment == "bearish":
                overall_sentiment_score -= score

        # Impact level
        impact = self._determine_impact(text)

        # Overall sentiment
        if overall_sentiment_score > 0.3:
            overall = "bullish"
        elif overall_sentiment_score < -0.3:
            overall = "bearish"
        else:
            overall = "neutral"

        return {
            "currencies": currency_sentiment,
            "overall_sentiment": overall,
            "impact": impact,
            "confidence": round(abs(overall_sentiment_score), 2),
            "text_length": len(text),
        }

    def _determine_impact(self, text: str) -> str:
        """Determine impact level from text content."""
        high_hits = sum(1 for trigger in HIGH_IMPACT_TRIGGERS if trigger in text)
        if high_hits >= 2:
            return "high"
        elif high_hits == 1:
            return "medium"
        return "low"


def _currency_aliases(currency: str) -> str:
    """Return common aliases for currencies."""
    aliases = {
        "USD": "dollar",
        "EUR": "euro",
        "GBP": "pound sterling",
        "JPY": "yen",
        "XAU": "gold",
    }
    return aliases.get(currency, currency.lower())


def aggregate_sentiment(classifications: list) -> dict:
    """
    Aggregate multiple news classifications into a single currency sentiment view.

    Args:
        classifications: List of classification dicts.

    Returns:
        dict: Currency-level aggregated sentiment.
    """
    currency_data = defaultdict(lambda: {"bullish": 0, "bearish": 0, "neutral": 0,
                                          "total": 0, "confidence_sum": 0.0,
                                          "high_impact": 0})

    for c in classifications:
        for currency, data in c.get("currencies", {}).items():
            sent = data.get("sentiment", "neutral")
            currency_data[currency][sent] += 1
            currency_data[currency]["total"] += 1
            currency_data[currency]["confidence_sum"] += data.get("confidence", 0)
            if c.get("impact") == "high":
                currency_data[currency]["high_impact"] += 1

    result = {}
    for currency, data in currency_data.items():
        total = max(data["total"], 1)
        if data["bullish"] > data["bearish"]:
            sentiment = "bullish"
        elif data["bearish"] > data["bullish"]:
            sentiment = "bearish"
        else:
            sentiment = "neutral"

        avg_conf = data["confidence_sum"] / total
        impact_score = min(1.0, data["high_impact"] * 0.3)

        # Risk level
        high_pct = data["high_impact"] / total
        if high_pct > 0.3 and sentiment == "bearish":
            risk = "high"
        elif high_pct > 0.1:
            risk = "medium"
        else:
            risk = "low"

        result[currency] = {
            "sentiment": sentiment,
            "impact_score": round(impact_score, 2),
            "active_events": [],
            "risk_level": risk,
            "article_count": data["total"],
        }

    return result