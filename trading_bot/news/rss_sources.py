"""
RSS news feed sources for macro-economic data.

Integrates free RSS feeds from multiple sources.
All feeds are read-only — no POST requests, no scraping against TOS.
Built-in timeouts and error handling for reliability.
"""

import re
import time
import html
from datetime import datetime, timedelta
import xml.etree.ElementTree as ET
from typing import Optional
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

from trading_bot.utils.logger import logger


# RSS feed definitions (all free, no API key required)
RSS_FEEDS = {
    "yahoo_finance_macro": "https://finance.yahoo.com/news/rssindex",
    "fxstreet_forex": "https://www.fxstreet.com/feed/news",
    "investing_com_news": "https://www.investing.com/rss/news.rss",
    "forexfactory_calendar": "https://www.forexfactory.com/calendar.xml",
}


def fetch_feed(url: str, timeout: int = 15, max_retries: int = 2) -> Optional[str]:
    """Fetch an RSS feed with timeout and retry."""
    for attempt in range(1, max_retries + 1):
        try:
            req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(req, timeout=timeout) as resp:
                data = resp.read().decode("utf-8", errors="ignore")
                logger.debug(f"Fetched {url}: {len(data)} bytes")
                return data
        except HTTPError as e:
            if attempt < max_retries:
                wait = attempt * 2
                logger.warning(f"HTTP {e.code} fetching {url}, retry in {wait}s")
                time.sleep(wait)
            else:
                logger.warning(f"Failed to fetch {url}: HTTP {e.code}")
        except URLError as e:
            if attempt < max_retries:
                wait = attempt * 2
                logger.warning(f"URL error {url}: {e.reason}, retry in {wait}s")
                time.sleep(wait)
            else:
                logger.warning(f"Failed to fetch {url}: {e.reason}")
        except Exception as e:
            logger.warning(f"Exception fetching {url}: {e}")
            break
    return None


def parse_rss_items(xml_data: str, source: str) -> list:
    """Parse RSS XML into list of news items."""
    items = []
    try:
        root = ET.fromstring(xml_data)
        # RSS feeds use channel -> item
        for item in root.iter("item"):
            title = _get_text(item, "title")
            link = _get_text(item, "link")
            desc = _get_text(item, "description")
            pub_date_str = _get_text(item, "pubDate")
            pub_date = _parse_date(pub_date_str) if pub_date_str else datetime.now()

            items.append({
                "source": source,
                "title": html.unescape(title) if title else "",
                "description": html.unescape(desc) if desc else "",
                "link": link or "",
                "published": pub_date,
                "fetched_at": datetime.now(),
            })
    except ET.ParseError as e:
        logger.warning(f"XML parse error for {source}: {e}")
    return items


def _get_text(parent, tag: str) -> Optional[str]:
    """Safe text extraction from XML element."""
    el = parent.find(tag)
    return el.text.strip() if el is not None and el.text else None


def _parse_date(date_str: str) -> Optional[datetime]:
    """Parse common RSS date formats."""
    formats = [
        "%a, %d %b %Y %H:%M:%S %z",
        "%a, %d %b %Y %H:%M:%S %Z",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%d %H:%M:%S",
    ]
    for fmt in formats:
        try:
            parsed = datetime.strptime(date_str, fmt)
            if parsed.tzinfo:
                parsed = parsed.replace(tzinfo=None)
            return parsed
        except ValueError:
            continue
    return datetime.now()


def fetch_all_feeds() -> list:
    """Fetch and parse all configured RSS feeds."""
    all_items = []
    for name, url in RSS_FEEDS.items():
        logger.info(f"Fetching RSS feed: {name}")
        data = fetch_feed(url)
        if data:
            items = parse_rss_items(data, name)
            logger.info(f"  Got {len(items)} items from {name}")
            all_items.extend(items)
        else:
            logger.warning(f"  No data from {name}")
    return all_items


def filter_relevant(items: list, currencies: list = None) -> list:
    """
    Filter news items relevant to our trading currencies.

    Args:
        items: List of news item dicts.
        currencies: List of currency keywords (default: USD, EUR, GBP, JPY, XAU).

    Returns:
        Filtered list with relevance score.
    """
    if currencies is None:
        currencies = ["USD", "EUR", "GBP", "JPY", "XAU", "Gold", "Forex", "Fed", "ECB", "BOJ", "CPI", "NFP"]

    relevant = []
    for item in items:
        text = (item.get("title", "") + " " + item.get("description", "")).lower()
        matched = [c for c in currencies if c.lower() in text]
        if matched:
            item["matched_currencies"] = matched
            item["relevance_score"] = len(matched) / max(len(currencies), 1)
            relevant.append(item)

    relevant.sort(key=lambda x: x["relevance_score"], reverse=True)
    return relevant