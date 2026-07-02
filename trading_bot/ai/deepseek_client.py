"""
DeepSeek V4 Pro AI integration client.

Modified to accept optional news_context for enhanced macro awareness.
News context does NOT change the AI's role — it only provides additional
data for sentiment analysis.
"""

import json
import time
import requests
from trading_bot.utils.logger import logger
from trading_bot.config import Config

DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
DEFAULT_TIMEOUT = 30
MAX_RETRIES = 3
RETRY_DELAY = 2

SYSTEM_PROMPT = (
    "You are a financial market sentiment analysis assistant. "
    "Your role is to analyze market data, technical indicators, rule engine outputs, "
    "and news context. "
    "You must ONLY provide sentiment analysis, confidence scoring, "
    "contradiction detection, and risk flagging. "
    "You must NEVER output trading signals (BUY/SELL) or trade execution instructions. "
    "Always respond with valid JSON only, no additional text."
)


class DeepSeekClient:
    def __init__(self, api_key: str = ""):
        self.api_key = api_key or Config.DEEPSEEK_API_KEY
        self._initialized = False
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        })
        logger.info("DeepSeekClient created.")

    def initialize(self) -> bool:
        if self._initialized:
            return True
        if not self.api_key:
            logger.warning("DeepSeek API key is empty.")
            self._initialized = False
            return False
        try:
            resp = self._session.get("https://api.deepseek.com/v1/models", timeout=10)
            if resp.status_code == 200:
                self._initialized = True
                logger.info("DeepSeekClient initialized.")
                return True
            else:
                logger.error(f"DeepSeek check failed: HTTP {resp.status_code}")
                return False
        except Exception as e:
            logger.error(f"DeepSeek check failed: {e}")
            return False

    def analyze_market(self, market_payload: dict) -> dict:
        if not self._initialized:
            return self._fallback("AI client not initialized")

        messages = self._build_messages(market_payload)

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = self._session.post(
                    DEEPSEEK_API_URL,
                    json={
                        "model": "deepseek-chat",
                        "messages": messages,
                        "temperature": 0.3,
                        "max_tokens": 500,
                        "response_format": {"type": "json_object"},
                    },
                    timeout=DEFAULT_TIMEOUT,
                )
                if resp.status_code == 200:
                    return self._parse(resp.json())
                elif resp.status_code == 429:
                    retry = int(resp.headers.get("Retry-After", RETRY_DELAY))
                    logger.warning(f"Rate limited. Retry in {retry}s")
                    time.sleep(retry)
                else:
                    logger.error(f"API error {resp.status_code}: {resp.text[:200]}")
                    if attempt < MAX_RETRIES:
                        time.sleep(RETRY_DELAY)
            except requests.Timeout:
                logger.warning(f"Timeout (attempt {attempt})")
                time.sleep(RETRY_DELAY)
            except Exception as e:
                logger.error(f"Request failed: {e}")
                time.sleep(RETRY_DELAY)

        return self._fallback("API unavailable after retries")

    def _build_messages(self, payload: dict) -> list:
        payload_str = json.dumps(payload, indent=2)
        user = (
            "Analyze the following market data and return a structured JSON analysis.\n\n"
            "```json\n" + payload_str + "\n```\n\n"
            "Respond with ONLY valid JSON:\n"
            '{"sentiment":"bullish|bearish|neutral","confidence":0-100,'
            '"reasoning":"short explanation","risk_flag":"low|medium|high",'
            '"conflicts_detected":true|false}'
        )
        return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]

    def _parse(self, api_resp: dict) -> dict:
        try:
            choices = api_resp.get("choices", [])
            if not choices:
                return self._fallback("Empty response")
            content = choices[0].get("message", {}).get("content", "")
            analysis = json.loads(content)
            required = {"sentiment", "confidence", "reasoning", "risk_flag", "conflicts_detected"}
            if not all(k in analysis for k in required):
                return self._fallback(f"Missing keys: {required - set(analysis.keys())}")
            sentiment = analysis.get("sentiment", "neutral").lower().strip()
            if sentiment not in ("bullish", "bearish", "neutral"):
                sentiment = "neutral"
            risk_flag = analysis.get("risk_flag", "medium").lower().strip()
            if risk_flag not in ("low", "medium", "high"):
                risk_flag = "medium"
            confidence = max(0, min(100, int(analysis.get("confidence", 50))))
            return {
                "sentiment": sentiment,
                "confidence": confidence,
                "reasoning": str(analysis.get("reasoning", "")),
                "risk_flag": risk_flag,
                "conflicts_detected": bool(analysis.get("conflicts_detected", False)),
                "ai_unavailable": False,
            }
        except Exception as e:
            logger.error(f"Parse error: {e}")
            return self._fallback(f"Parse error: {e}")

    def _fallback(self, reason: str = "") -> dict:
        return {
            "sentiment": "neutral",
            "confidence": 0,
            "reasoning": f"AI unavailable: {reason}" if reason else "AI unavailable",
            "risk_flag": "medium",
            "conflicts_detected": False,
            "ai_unavailable": True,
        }

    def shutdown(self):
        self._session.close()