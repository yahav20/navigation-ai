"""Shared budget-resolution rule.

LLMs are unreliable at adding/subtracting large numbers (e.g. "add $10000" to a
$4000 budget has been observed to come back as $15000 instead of $14000, even
with a matching worked example in the prompt). To remove that failure mode,
the model only extracts the raw signed delta/percent the user stated — never
the computed total — and this function does the arithmetic deterministically.
"""


def resolve_budget(
    current: float | None,
    *,
    absolute: float | None,
    delta: float | None,
    delta_pct: float | None,
) -> float | None:
    """Compute the new budget from at most one of absolute/delta/delta_pct.

    Returns None when none apply (nothing for the caller to update).
    """
    if absolute is not None:
        return absolute
    if delta is not None:
        return (current or 0) + delta
    if delta_pct is not None and current is not None:
        return current * (1 + delta_pct / 100)
    return None
