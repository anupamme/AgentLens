"""Privacy validation module for AgentLens.

Implements three experiments to validate privacy preservation:
1. PII leakage testing
2. Re-identification attacks
3. Utility-privacy trade-off analysis
"""

from agentlens.privacy.leakage_test import PIILeakageReport, PIILeakageTest
from agentlens.privacy.pii_generator import PIIGenerator, SyntheticPII
from agentlens.privacy.reidentification_test import (
    MockAdversary,
    ReidentificationResult,
    ReidentificationTest,
)
from agentlens.privacy.runner import run_full_privacy_validation
from agentlens.privacy.utility_tradeoff import UtilityPrivacyAnalysis, UtilityPrivacyReport

__all__ = [
    "PIIGenerator",
    "SyntheticPII",
    "PIILeakageTest",
    "PIILeakageReport",
    "ReidentificationResult",
    "ReidentificationTest",
    "MockAdversary",
    "UtilityPrivacyAnalysis",
    "UtilityPrivacyReport",
    "run_full_privacy_validation",
]
