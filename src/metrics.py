"""
Métricas de performance por padrão + validação estatística.
Sem indicadores — apenas estatística sobre os resultados (target/stop/timeout)
gerados em patterns.py a partir de candles reais.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_metrics(outcomes: pd.DataFrame, min_occurrences: int = 500) -> dict | None:
    """Calcula taxa de acerto, profit factor, expectância, MAE/MFE médios,
    IC 95% da taxa de acerto (Wilson) e p-valor (bootstrap contra expectância 0)."""
    n = len(outcomes)
    if n < min_occurrences:
        return {"n": n, "insufficient_sample": True}

    wins = outcomes[outcomes["result"] == "target"]
    losses = outcomes[outcomes["result"] == "stop"]
    timeouts = outcomes[outcomes["result"] == "timeout"]

    win_rate = len(wins) / n
    gross_win = (wins["reward"]).sum()
    gross_loss = (losses["risk"]).sum()
    profit_factor = gross_win / gross_loss if gross_loss > 0 else np.inf

    r_multiples = np.where(
        outcomes["result"] == "target", outcomes["rr_planned"],
        np.where(outcomes["result"] == "stop", -1.0, 0.0),
    )
    expectancy = r_multiples.mean()

    # IC 95% (Wilson score) para taxa de acerto
    z = 1.96
    p = win_rate
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = (z * np.sqrt((p * (1 - p) / n) + (z**2 / (4 * n**2)))) / denom
    ci_low, ci_high = center - half, center + half

    # p-valor via bootstrap: H0 = expectância <= 0
    rng = np.random.default_rng(42)
    boot_means = []
    r = r_multiples
    for _ in range(2000):
        sample = rng.choice(r, size=n, replace=True)
        boot_means.append(sample.mean())
    boot_means = np.array(boot_means)
    p_value = (boot_means <= 0).mean()

    return {
        "n": n,
        "win_rate": win_rate,
        "win_rate_ci95": (ci_low, ci_high),
        "profit_factor": profit_factor,
        "expectancy_R": expectancy,
        "p_value_bootstrap": p_value,
        "mae_mean": outcomes["mae"].mean(),
        "mfe_mean": outcomes["mfe"].mean(),
        "bars_to_outcome_mean": outcomes["bars"].mean(),
        "rr_planned_mean": outcomes["rr_planned"].mean(),
        "timeout_rate": len(timeouts) / n,
        "insufficient_sample": False,
    }


def validate_train_val_test(outcomes_by_period: dict[str, pd.DataFrame], min_occurrences: int = 500,
                             min_profit_factor: float = 1.30, max_p_value: float = 0.05) -> dict:
    """Calcula métricas em treino (2024) / validação (2025) / teste (2026)
    e só aprova o padrão se for consistente e significativo nas três fases."""
    metrics = {period: compute_metrics(df, min_occurrences=min_occurrences) for period, df in outcomes_by_period.items()}

    approved = True
    reasons = []
    for period, m in metrics.items():
        if m is None or m.get("insufficient_sample"):
            approved = False
            reasons.append(f"{period}: amostra insuficiente (n={m['n'] if m else 0} < {min_occurrences})")
            continue
        if m["profit_factor"] < min_profit_factor:
            approved = False
            reasons.append(f"{period}: profit factor {m['profit_factor']:.2f} < {min_profit_factor}")
        if m["expectancy_R"] <= 0:
            approved = False
            reasons.append(f"{period}: expectância não positiva ({m['expectancy_R']:.3f}R)")
        if m["p_value_bootstrap"] >= max_p_value:
            approved = False
            reasons.append(f"{period}: p-valor {m['p_value_bootstrap']:.3f} >= {max_p_value}")

    # estabilidade: expectância e win rate não podem variar demais entre fases
    valid_metrics = [m for m in metrics.values() if m and not m.get("insufficient_sample")]
    if len(valid_metrics) == 3:
        exps = [m["expectancy_R"] for m in valid_metrics]
        wrs = [m["win_rate"] for m in valid_metrics]
        if (max(exps) - min(exps)) > 0.5:
            approved = False
            reasons.append(f"expectância instável entre fases: {exps}")
        if (max(wrs) - min(wrs)) > 0.20:
            approved = False
            reasons.append(f"taxa de acerto instável entre fases: {wrs}")

    return {"approved": approved, "reasons": reasons, "metrics": metrics}
