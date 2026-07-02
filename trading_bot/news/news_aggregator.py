"""
News Aggregator — master controller for the news intelligence layer.

Collects data from RSS feeds and economic calendar, classifies sentiment,
aggregates results, and provides structured output for the risk engine.

Designed to run independently and fail gracefully if sources are unavailable.
"""

import time
from datetime import datetime, timedelta
from typing import Optional

from trading_bot.utils.logger import logger
from trading_bot.news.rss_sources import fetch_all_feeds, filter_relevant
from trading_bot.news.economic_calendar import EconomicCalendar
from trading_bot.news.sentiment_engine import SentimentEngine, aggregate_sentiment
from trading_bot.news.event_filter import filter_events


class NewsAggregator:
    """
    Master news controller.

    Collects, classifies, and stores market news.
    Provides a structured snapshot for the risk engine.
    """

    def __init__(self, update_interval_minutes: int = 15):
        self.update_interval = timedelta(minutes=update_interval_minutes)
        self.last_update: Optional[datetime] = None
        self.calendar = EconomicCalendar()
        self.sentiment = SentimentEngine()
        self._news_buffer: list = []
        self._currency_sentiment: dict = {}
        self._global_risk_mode: str = "low"
        self._next_event: Optional[dict] = None
        self._minutes_to_event: Optional[int] = None

    def update(self, force: bool = False):
        """Fetch and process news data if enough time has passed since last update."""
        now = datetime.now()
        if not force and self.last_update and (now - self.last_update) < self.update_interval:
            logger.debug("News aggregator: skipping update (within interval)")
            return

        logger.info("News aggregator: starting update cycle")

        # Step 1: Economic calendar
        try:
            self.calendar.update()
        except Exception as e:
            logger.warning(f"Calendar update failed: {e}")

        # Step 2: Fetch RSS feeds
        raw_items = []
        try:
            raw_items = fetch_all_feeds()
            if raw_items:
                relevant = filter_relevant(raw_items)
                logger.info(f"  Relevant items: {len(relevant)}")
            else:
                relevant = []
                logger.info("  No RSS data available (offline or blocked)")
        except Exception as e:
            logger.warning(f"RSS fetch failed: {e}")
            relevant = []

        # Step 3: Classify sentiment
        classifications = []
        for item in relevant:
            try:
                cls = self.sentiment.classify(item)
                classifications.append(cls)
            except Exception as e:
                logger.debug(f"Classification error: {e}")

        # Step 4: Filter and store
        if relevant and classifications:
            filtered = filter_events(relevant, classifications)
            self._news_buffer = filtered[-50:]  # keep last 50
        else:
            self._news_buffer = []

        # Step 5: Aggregate currency sentiment
        if classifications:
            self._currency_sentiment = aggregate_sentiment(classifications)
        else:
            self._currency_sentiment = {}

        # Step 6: Store calendar-derived risk data
        self._global_risk_mode = self.calendar.get_global_risk_mode()
        self._next_event = self.calendar.get_next_high_impact()
        self._minutes_to_event = self.calendar.minutes_to_next_event()

        self.last_update = now
        logger.info(f"News aggregator: update complete | risk_mode={self._global_risk_mode} "
                     f"| events={len(self.calendar.events_buffer)} | news={len(self._news_buffer)}")

    def get_news_context(self) -> dict:
        """
        Get the current news context for injection into AI payload.

        Returns structured dict:
        {
            "currency_news": {...},
            "global_risk_mode": "low|medium|high|news_blackout",
            "next_high_impact_event": "...",
            "time_to_event_minutes": N or None,
            "news_items_count": N,
        }
        """
        next_event_name = self._next_event.get("name", "N/A") if self._next_event else None

        return {
            "currency_news": self._currency_sentiment,
            "global_risk_mode": self._global_risk_mode,
            "next_high_impact_event": next_event_name,
            "time_to_event_minutes": self._minutes_to_event,
            "news_items_count": len(self._news_buffer),
        }

    def get_risk_overlay(self) -> dict:
        """
        Get risk overlay values for the risk engine.

        Returns:
            dict: {
                "news_block_all_trades": bool,
                "reduce_lot_by_percent": float (0.0-0.5),
                "increase_risk_score_by": int (0-20),
                "reason": str,
            }
        """
        result = {
            "news_block_all_trades": False,
            "reduce_lot_by_percent": 0.0,
            "increase_risk_score_by": 0,
            "reason": "No news overlay",
        }

        # Blackout — block all trades
        if self._global_risk_mode == "news_blackout":
            result["news_block_all_trades"] = True
            result["reason"] = "News blackout — high-impact event active window"
            return result

        # High risk — reduce position and increase score
        if self._global_risk_mode == "high":
            result["reduce_lot_by_percent"] = 0.5
            result["increase_risk_score_by"] = 20
            result["reason"] = "High-impact event within 30 minutes"

        # Medium risk — moderate adjustment
        elif self._global_risk_mode == "medium":
            result["reduce_lot_by_percent"] = 0.25
            result["increase_risk_score_by"] = 10
            result["reason"] = "Medium-risk news environment"

        # Check for conflicting news sentiment
        if self._currency_sentiment:
            sentiments = {v["sentiment"] for v in self._currency_sentiment.values()}
            if len(sentiments) > 1 and "bearish" in sentiments and "bullish" in sentiments:
                result["increase_risk_score_by"] += 10
                result["reason"] += " + conflicting news sentiment"

        return result

    def summary(self) -> dict:
        """Get a full summary for logging."""
        return {
            "global_risk_mode": self._global_risk_mode,
            "next_event": self._next_event,
            "minutes_to_event": self._minutes_to_event,
            "news_count": len(self._news_buffer),
            "currencies_tracked": list(self._currency_sentiment.keys()),
            "last_update": str(self.last_update) if self.last_update else None,
        }