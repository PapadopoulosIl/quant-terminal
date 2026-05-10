"""
core_analysis.py — Unified Analysis Abstraction Layer
=====================================================
The single source of truth for all analysis results.
Ensures consistency between Risk, Advisory, and Valuation layers.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List
from abc import ABC, abstractmethod
import pandas as pd

# 1. Common Enums
class MarketRegime(Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"

class RiskLevel(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class SignalStrength(Enum):
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    NEUTRAL = "NEUTRAL"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"

# 2. Central Data Model
@dataclass
class UnifiedAnalysisResult:
    ticker: str
    market_regime: MarketRegime
    risk_level: RiskLevel
    overall_signal: SignalStrength
    confidence_score: float          # 0.0 - 1.0
    key_drivers: List[str] = field(default_factory=list)
    hedge_recommendation: Optional[str] = None
    macro_context: Optional[str] = None
    timestamp: str = ""

    # --- ΝΕΑ ΠΕΔΙΑ ΓΙΑ ΕΞΗΓΗΣΙΜΟΤΗΤΑ (Explainable AI) ---
    plain_summary: str = ""             # 1-2 προτάσεις στα ελληνικά
    why_this_signal: str = ""           # Η πιο σημαντική αιτιολογία
    user_friendly_risk: str = ""        # π.χ. "1 στις 7 πιθανότητες για μεγάλη πτώση"
    recommended_action: str = ""        # "Μείωσε θέση κατά 30%"
    confidence_level: str = ""          # "Υψηλή (89%)"

# 3. Abstract Base Class
class AnalysisComponent(ABC):
    @abstractmethod
    def analyze(self, data: dict) -> UnifiedAnalysisResult:
        """Each component must implement this to ensure consistent output."""
        pass
