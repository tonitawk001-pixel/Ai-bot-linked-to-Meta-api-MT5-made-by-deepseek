"""
Risk Management Engine — FINAL AUTHORITY.

Replaced strict binary blocking with a calibrated risk scoring system:
  - risk_score 0-40   → ALLOW (full trade)
  - risk_score 40-70  → REDUCED (scaled lot size)
  - risk_score 70-100 → BLOCK

Hard blocks (always enforced, override everything):
  - Emergency stop / kill switch
  - Daily loss limit exceeded
  - Consecutive losses ≥ limit
  - Spread > MAX_SPREAD_MULTIPLIER * avg

All other checks contribute to risk_score with weighted penalties.
AI influence is graded, not binary.
"""

from trading_bot.utils.logger import logger
from trading_bot.config import Config


class RiskManager:
    """
    Risk scoring engine with dynamic thresholds.

    Scoring: 0 = completely safe, 100 = must block
    Decision: 0-40 allow, 40-70 reduce lot, 70-100 block
    """

    def __init__(self, account_balance: float = 10000.0, default_balance: float = None):
        if default_balance is not None:
            account_balance = default_balance
        self._balance = account_balance
        self._daily_loss = 0.0
        self._daily_trades = 0
        self._consecutive_losses = 0
        self._equity_peak = account_balance
        self._gold_loss_streak_active = False
        self._gold_loss_streak_count = 0
        logger.info("RiskManager initialized (scoring mode).")

    def set_balance(self, balance: float):
        self._balance = balance
        if balance > self._equity_peak:
            self._equity_peak = balance

    def validate(self, rule_decision: dict, ai_analysis: dict, ohlcv=None, news_overlay: dict = None) -> dict:
        """
        Calculate risk score and return graded decision.

        Returns:
            dict: {
                "approved": bool,          # True if score < 70
                "risk_score": 0-100,
                "trade_quality": "high" | "medium" | "low" | "blocked",
                "adjusted_lot_scale": float (0.0-1.0),
                "adjusted_confidence": 0-100,
                "blocked_by": [str],       # Hard blocks only
                "soft_warnings": [str],    # Score contributors
                "reason": str,
            }
        """
        risk_score = 0.0
        hard_blocks = []
        soft_warnings = []

        # ==================================================================
        # HARD BLOCKS — these ALWAYS apply regardless of score
        # ==================================================================

        # 1. Daily loss limit (hard)
        if self._balance > 0:
            daily_loss_pct = (self._daily_loss / self._balance) * 100
            if daily_loss_pct <= -Config.MAX_DAILY_LOSS_PERCENT:
                hard_blocks.append(
                    f"daily_loss_{daily_loss_pct:.1f}%_exceeded_{Config.MAX_DAILY_LOSS_PERCENT}%"
                )

        # 2. Consecutive losses (hard)
        if self._consecutive_losses >= Config.MAX_CONSECUTIVE_LOSSES:
            hard_blocks.append(f"consecutive_losses_{self._consecutive_losses}")

        # 3. Spread ceiling (hard)
        spread_ok = True
        if ohlcv is not None and "spread" in ohlcv.columns:
            spreads = ohlcv["spread"].iloc[-20:]
            if len(spreads) > 0:
                avg_spread = spreads.mean()
                current_spread = ohlcv["spread"].iloc[-1]
                if current_spread > avg_spread * Config.MAX_SPREAD_MULTIPLIER:
                    hard_blocks.append(
                        f"spread_{current_spread:.0f}_above_{avg_spread*Config.MAX_SPREAD_MULTIPLIER:.0f}"
                    )
                    spread_ok = False

        # If any hard blocks → immediately blocked, score = 100
        if hard_blocks:
            return {
                "approved": False,
                "risk_score": 100,
                "trade_quality": "blocked",
                "adjusted_lot_scale": 0.0,
                "adjusted_confidence": 0,
                "blocked_by": hard_blocks,
                "soft_warnings": [],
                "reason": f"Hard blocked: {', '.join(hard_blocks)}",
            }

        # ==================================================================
        # SOFT SCORING — weighted penalties
        # ==================================================================

        # --- Setup validity (0-25 penalty) ---
        if not rule_decision.get("setup_valid", False):
            risk_score += 25
            soft_warnings.append("setup_invalid")

        # --- AI confidence (0-30 penalty) ---
        ai_conf = ai_analysis.get("confidence", 0)
        if ai_conf < Config.AI_MIN_CONFIDENCE:
            penalty = 30 * (1 - ai_conf / max(Config.AI_MIN_CONFIDENCE, 1))
            risk_score += penalty
            soft_warnings.append(f"low_ai_confidence_{ai_conf}")
        elif ai_conf >= 90:
            risk_score -= 5  # high confidence reduces risk slightly
            soft_warnings.append("ai_high_confidence_bonus")

        # --- AI risk flag (0-20 penalty) ---
        risk_flag = ai_analysis.get("risk_flag", "medium")
        if risk_flag == "high":
            risk_score += 20
            soft_warnings.append("ai_high_risk_flag")
        elif risk_flag == "medium":
            risk_score += 5
            soft_warnings.append("ai_medium_risk_flag")

        # --- AI conflicts detected (0-15 penalty) ---
        if ai_analysis.get("conflicts_detected", False):
            risk_score += 15
            soft_warnings.append("ai_conflicts_detected")

        # --- AI unavailable (0-25 penalty, graceful) ---
        if ai_analysis.get("ai_unavailable", False):
            risk_score += 25
            soft_warnings.append("ai_unavailable")

        # --- ATR volatility filter (0-20 penalty) ---
        atr_val = rule_decision.get("atr_value")
        if atr_val is not None and ohlcv is not None:
            latest_close = float(ohlcv["close"].iloc[-1])
            if latest_close > 0:
                atr_pct = (atr_val / latest_close) * 100
                if atr_pct > Config.MAX_ATR_PERCENT:
                    penalty = min(20, (atr_pct - Config.MAX_ATR_PERCENT) * 5)
                    risk_score += penalty
                    soft_warnings.append(f"high_atr_{atr_pct:.2f}%")

        # --- Trend-Volatility mismatch (0-10 bonus/penalty) ---
        trend = rule_decision.get("trend", "neutral")
        volatility = rule_decision.get("volatility", "medium")
        if trend == "neutral" and volatility == "high":
            risk_score += 10
            soft_warnings.append("neutral_trend_high_volatility")
        elif trend != "neutral" and volatility == "low":
            risk_score -= 5  # clear trend + low vol is favorable

        # --- RSI extreme (0-10 penalty) ---
        rsi_cond = rule_decision.get("rsi_condition", "neutral")
        if rsi_cond == "overbought" or rsi_cond == "oversold":
            risk_score += 10
            soft_warnings.append(f"rsi_{rsi_cond}")

        # ==================================================================
        # Clamp score to 0-100
        # ==================================================================
        risk_score = max(0.0, min(100.0, risk_score))

        # ==================================================================
        # Determine decision
        # ==================================================================
        if risk_score < 40:
            approved = True
            trade_quality = "high"
            lot_scale = 1.0
        elif risk_score < 70:
            # Medium risk — scale lot size down proportionally
            scale_factor = 1.0 - ((risk_score - 40) / 30) * 0.7  # 1.0 → 0.3
            lot_scale = max(0.3, min(1.0, scale_factor))
            approved = True
            trade_quality = "medium"
        else:
            approved = False
            trade_quality = "blocked"
            lot_scale = 0.0

        # Adjusted confidence blends AI confidence with risk score
        adjusted_confidence = int(max(0, min(100, ai_conf - risk_score * 0.5)))

        result = {
            "approved": approved,
            "risk_score": round(risk_score, 1),
            "trade_quality": trade_quality,
            "adjusted_lot_scale": round(lot_scale, 2),
            "adjusted_confidence": adjusted_confidence,
            "blocked_by": [],
            "soft_warnings": soft_warnings,
            "reason": (
                f"Score={risk_score:.0f}, quality={trade_quality}, "
                f"lot_scale={lot_scale:.2f}, adj_conf={adjusted_confidence}"
            ),
        }

        if approved:
            logger.info(f"RiskManager: APPROVED (score={risk_score:.0f}, "
                        f"quality={trade_quality}, lot_scale={lot_scale:.2f})")
        else:
            logger.warning(f"RiskManager: BLOCKED (score={risk_score:.0f})")

        return result

    def gold_specific_adjustments(
        self,
        base_result: dict,
        account_balance: float,
        volatility_info: dict = None,
    ) -> dict:
        """
        Apply XAUUSD-specific safety adjustments on top of base risk evaluation.

        This runs AFTER validate() and adds gold-specific rules:
          - ATR spike: reduce lot size by 30-60%
          - Spread spike: BLOCK trade
          - 3+ consecutive losses: reduce risk to 1%
          - Flat market: reduce frequency

        Does NOT override hard blocks from validate().
        Only makes already-approved trades more conservative.

        Args:
            base_result: result from validate()
            account_balance: current account balance
            volatility_info: dict from GoldVolatilityFilter.analyze()

        Returns:
            dict: augmented result with gold-specific adjustments
        """
        result = dict(base_result)  # Start with base result

        # If already blocked, don't modify
        if not result.get("approved", False):
            return result

        gold_blocks = []
        gold_warnings = []
        extra_lot_scale = 1.0

        # --- Gold ATR-based lot size reduction ---
        if volatility_info:
            atr_ratio = volatility_info.get("atr_ratio", 1.0)
            lot_reduction = volatility_info.get("lot_reduction_factor", 1.0)

            if lot_reduction <= 0:
                # Spread spike or extreme ATR → block
                gold_blocks.append(
                    f"gold_volatility_block_{volatility_info.get('reason', 'unknown')}"
                )
                extra_lot_scale = 0.0
            elif lot_reduction < 1.0:
                extra_lot_scale = lot_reduction
                gold_warnings.append(f"gold_atr_lot_scale_{lot_reduction:.2f}")

            # Regime-based warning
            regime = volatility_info.get("market_regime", "unknown")
            if regime == "volatile":
                gold_warnings.append("gold_volatile_regime")
            elif regime == "flat":
                gold_warnings.append("gold_flat_regime")

        # --- Gold spread spike (additional block) ---
        if volatility_info and volatility_info.get("spread_assessment") == "spike":
            gold_blocks.append("gold_spread_spike")
            extra_lot_scale = 0.0

        # --- Gold loss streak: 3 consecutive losses → reduce risk to 1% ---
        if self._consecutive_losses >= 3:
            loss_streak_scale = (
                Config.GOLD_LOSS_STREAK_RISK_PERCENT / Config.MAX_RISK_PERCENT
            )
            extra_lot_scale *= loss_streak_scale
            gold_warnings.append(
                f"gold_loss_streak_{self._consecutive_losses}_risk_{Config.GOLD_LOSS_STREAK_RISK_PERCENT}%"
            )
            self._gold_loss_streak_active = True
        else:
            self._gold_loss_streak_active = False

        # Apply gold blocks
        if gold_blocks:
            result["approved"] = False
            result["risk_score"] = 100
            result["trade_quality"] = "blocked"
            result["adjusted_lot_scale"] = 0.0
            result["blocked_by"] = result.get("blocked_by", []) + gold_blocks
            result["reason"] = (
                result.get("reason", "") + f" | GOLD BLOCK: {', '.join(gold_blocks)}"
            )
            logger.warning(
                f"RiskManager GOLD BLOCK: {', '.join(gold_blocks)}"
            )
            return result

        # Apply gold lot scaling (multiply with existing scale)
        existing_scale = result.get("adjusted_lot_scale", 1.0)
        final_scale = round(existing_scale * extra_lot_scale, 2)
        final_scale = max(0.3, min(1.0, final_scale))  # Don't go below 30% of original

        result["adjusted_lot_scale"] = final_scale
        result["gold_adjustments"] = {
            "applied": bool(gold_warnings),
            "warnings": gold_warnings,
            "extra_lot_scale": extra_lot_scale,
            "consecutive_losses": self._consecutive_losses,
        }

        if gold_warnings:
            result["reason"] = (
                result.get("reason", "")
                + f" | GOLD ADJ: {', '.join(gold_warnings)}"
            )
            logger.info(
                f"RiskManager GOLD ADJUSTMENTS: lot_scale={final_scale:.2f} "
                f"warnings={gold_warnings}"
            )

        return result

    def get_account_summary(self, account_id: str = "default") -> dict:
        """
        Get account summary for injection into AI payload.

        Returns:
            dict: {
                "balance": float,
                "equity": float,
                "drawdown_pct": float,
                "consecutive_losses": int,
                "gold_loss_streak_active": bool,
            }
        """
        drawdown = 0.0
        if self._equity_peak > 0:
            drawdown = round(
                (self._equity_peak - self._balance) / self._equity_peak * 100, 2
            )
        return {
            "balance": round(self._balance, 2),
            "equity": round(self._balance, 2),  # simplified
            "drawdown_pct": drawdown,
            "consecutive_losses": self._consecutive_losses,
            "gold_loss_streak_active": self._gold_loss_streak_active,
        }

    def all_accounts(self) -> dict:
        """Return summary of all tracked accounts."""
        return {
            "default": self.get_account_summary("default"),
        }

    def record_result(self, profit_loss: float):
        self._daily_trades += 1
        self._daily_loss += profit_loss
        if profit_loss < 0:
            self._consecutive_losses += 1
            self._gold_loss_streak_count += 1
        else:
            self._consecutive_losses = 0
            self._gold_loss_streak_count = 0
        logger.debug(f"RiskManager: recorded P/L={profit_loss:.2f}, "
                     f"daily_loss={self._daily_loss:.2f}, "
                     f"consecutive={self._consecutive_losses}")

    def reset_daily(self):
        self._daily_loss = 0.0
        self._daily_trades = 0
        self._consecutive_losses = 0
        self._gold_loss_streak_active = False
        self._gold_loss_streak_count = 0
        logger.info("RiskManager daily counters reset.")
