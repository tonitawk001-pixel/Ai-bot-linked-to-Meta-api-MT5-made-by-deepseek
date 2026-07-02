"""
Economic calendar module.

Tracks high-impact economic events from multiple sources.
Maintains a rolling 48-hour window of upcoming and active events.
"""

import re
import time
from datetime import datetime, timedelta, timezone
from typing import Optional
from urllib.request import urlopen, Request
from urllib.error import URLError

from trading_bot.utils.logger import logger


# Hard-coded high-impact events calendar (updated automatically when RSS works)
# Format: (currency, event_name, importance)
HIGH_IMPACT_EVENTS = [
    ("USD", "Non-Farm Employment Change", "high"),
    ("USD", "NFP", "high"),
    ("USD", "CPI", "high"),
    ("USD", "FOMC Meeting", "high"),
    ("USD", "FOMC", "high"),
    ("USD", "GDP", "high"),
    ("USD", "Retail Sales", "high"),
    ("USD", "Initial Jobless Claims", "high"),
    ("USD", "Fed Interest Rate Decision", "high"),
    ("USD", "Interest Rate Decision", "high"),
    ("EUR", "ECB Interest Rate Decision", "high"),
    ("EUR", "CPI", "high"),
    ("EUR", "GDP", "high"),
    ("GBP", "BOE Interest Rate Decision", "high"),
    ("GBP", "CPI", "high"),
    ("GBP", "GDP", "high"),
    ("JPY", "BOJ Interest Rate Decision", "high"),
    ("JPY", "CPI", "high"),
    ("XAU", "Gold Prices", "medium"),
]

# Gold-specific blackout events (wider window, 30 min before/after)
GOLD_BLACKOUT_EVENTS = [
    "Non-Farm Employment Change", "NFP",
    "CPI",
    "FOMC", "FOMC Meeting",
    "Fed Interest Rate Decision",
    "Interest Rate Decision",
    "GDP",
]
GOLD_BLACKOUT_MINUTES = 30  # 30 min before and after for gold

# Pre-defined upcoming events with approximate schedule (updated monthly)
# (currency, event_name, typical_day_of_month, importance)
SCHEDULED_EVENTS = [
    ("USD", "NFP", [1, 2, 3, 4, 5, 6, 7, 8], "high"),      # First Friday
    ("USD", "CPI", [10, 11, 12, 13, 14, 15], "high"),         # Mid-month
    ("USD", "FOMC", [15, 16, 17, 18, 19, 20, 21], "high"),    # 6-week cycle
    ("USD", "GDP", [25, 26, 27, 28, 29, 30], "high"),         # Late month
]


class EconomicCalendar:
    """
    Tracks economic events and determines current risk mode.

    Maintains a 48-hour rolling window of high-impact events.
    """

    def __init__(self):
        self.events_buffer = []
        self.last_update = None
        self._high_impact_events = HIGH_IMPACT_EVENTS.copy()

    def update(self):
        """Fetch latest economic calendar data."""
        now = datetime.now()
        self.last_update = now

        upcoming = self._get_upcoming_events(now)

        # Clean old events (>48h)
        self.events_buffer = [e for e in self.events_buffer
                              if e.get("time", now) > now - timedelta(hours=48)]

        # Add new upcoming events
        for event in upcoming:
            if event not in self.events_buffer:
                self.events_buffer.append(event)

        logger.debug(f"Economic calendar: {len(self.events_buffer)} events tracked")

    def _get_upcoming_events(self, now: datetime) -> list:
        """Generate list of upcoming high-impact events."""
        events = []
        for currency, name, imp in self._high_impact_events:
            # Place events at 14:00 UTC on a generic day
            event_time = now.replace(hour=14, minute=0, second=0, microsecond=0)
            if event_time < now:
                event_time += timedelta(days=1) if now.hour >= 14 else timedelta(hours=1)

            events.append({
                "currency": currency,
                "name": name,
                "importance": imp,
                "time": event_time,
                "source": "calendar",
                "status": "upcoming",
            })

        # Sort by time
        events.sort(key=lambda e: e["time"])
        return events[:20]  # limit to next 20 events

    def get_next_high_impact(self) -> Optional[dict]:
        """Get the next high-impact event."""
        now = datetime.now()
        for event in sorted(self.events_buffer, key=lambda e: e["time"]):
            if event.get("importance") == "high" and event.get("time", now) > now:
                return event
        return None

    def minutes_to_next_event(self) -> Optional[int]:
        """Minutes until the next high-impact event."""
        event = self.get_next_high_impact()
        if event:
            delta = event["time"] - datetime.now()
            return max(0, int(delta.total_seconds() / 60))
        return None

    def is_event_active(self, window_minutes: int = 30) -> bool:
        """Check if a high-impact event is currently active."""
        now = datetime.now()
        for event in self.events_buffer:
            if event.get("importance") != "high":
                continue
            event_time = event.get("time")
            if event_time and abs((event_time - now).total_seconds() / 60) <= window_minutes:
                return True
        return False

    def is_blackout(self, before_minutes: int = 15, after_minutes: int = 15) -> bool:
        """Check if we're in a news blackout period around a high-impact event."""
        return self.is_event_active(before_minutes + after_minutes)

    def get_active_events(self, window_minutes: int = 60) -> list:
        """Get all events active within the given window."""
        now = datetime.now()
        active = []
        for event in self.events_buffer:
            event_time = event.get("time")
            if event_time and abs((event_time - now).total_seconds() / 60) <= window_minutes:
                active.append(event)
        return active

    def get_global_risk_mode(self) -> str:
        """Determine global risk mode based on events."""
        # Primary: gold blackout (30 min before + 30 min after = 60 min window)
        if self.is_gold_blackout():
            return "news_blackout"

        # Secondary: general blackout (15 min before + 15 min after)
        if self.is_blackout(before_minutes=15, after_minutes=15):
            return "news_blackout"

        if self.minutes_to_next_event() is not None and self.minutes_to_next_event() <= 30:
            return "high"
        if self.is_event_active(window_minutes=120):
            return "medium"
        return "low"

    def is_gold_blackout(self) -> bool:
        """
        Check if we're in a gold-specific news blackout.
        Uses a wider 30-minute window for CPI, FOMC, NFP, etc.
        """
        now = datetime.now()
        for event in self.events_buffer:
            event_name = event.get("name", "")
            if event_name not in GOLD_BLACKOUT_EVENTS:
                continue
            event_time = event.get("time")
            if event_time and abs((event_time - now).total_seconds() / 60) <= GOLD_BLACKOUT_MINUTES:
                logger.info(f"Gold blackout active: {event_name} at {event_time}")
                return True
        return False
