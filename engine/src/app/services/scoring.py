"""Hybrid score + verdict (Article formulas 5 & 6).

Pure functions, no IO. Verbatim from the old engine/scorer.py.
"""


def compute_hybrid_score(semantic: float, logical: float, alpha: float) -> float:
    """score = alpha*S + (1-alpha)*L."""
    return alpha * semantic + (1.0 - alpha) * logical


def compute_verdict(
    semantic: float, logical: float, tau_s: float, tau_l: float
) -> bool:
    """V = I[S >= tau_S] * I[L >= tau_L]."""
    return semantic >= tau_s and logical >= tau_l
