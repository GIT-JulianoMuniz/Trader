"""
Orquestrador: dados -> estrutura de mercado -> padrões -> validação -> ranking.
"""
from __future__ import annotations

import pandas as pd

from .data_loader import load_ohlc_csv, split_by_year
from .market_structure import find_fractals, extract_swings, swing_amplitudes
from .patterns import PATTERN_REGISTRY, attach_outcomes
from .metrics import validate_train_val_test


def run_discovery(csv_path: str, symbol: str, fractal_left: int = 2, fractal_right: int = 2,
                   min_occurrences: int = 500, min_profit_factor: float = 1.30,
                   max_p_value: float = 0.05) -> dict:
    df = load_ohlc_csv(csv_path, symbol=symbol)
    periods = split_by_year(df)

    results = {}
    for pattern_name, detector in PATTERN_REGISTRY.items():
        outcomes_by_period = {}
        for period_name, period_df in periods.items():
            if period_df.empty:
                outcomes_by_period[period_name] = pd.DataFrame()
                continue
            fr = find_fractals(period_df, left=fractal_left, right=fractal_right)
            swings = extract_swings(fr)
            occurrences = detector(period_df, swings)
            outcomes = attach_outcomes(period_df, occurrences)
            outcomes_by_period[period_name] = outcomes

        validation = validate_train_val_test(
            outcomes_by_period, min_occurrences=min_occurrences,
            min_profit_factor=min_profit_factor, max_p_value=max_p_value,
        )
        results[pattern_name] = validation

    return results


def rank_patterns(results: dict) -> list[tuple[str, dict]]:
    """Ordena por Profit Factor (teste 2026) -> Expectância -> Frequência,
    considerando somente padrões aprovados nas três fases."""
    approved = [(name, r) for name, r in results.items() if r["approved"]]

    def sort_key(item):
        _, r = item
        test_m = r["metrics"].get("test") or {}
        return (
            -(test_m.get("profit_factor", 0) or 0),
            -(test_m.get("expectancy_R", 0) or 0),
            -(test_m.get("n", 0) or 0),
        )

    approved.sort(key=sort_key)
    return approved


def format_report(results: dict, symbol: str) -> str:
    lines = [f"# Relatório de Descoberta de Padrões — {symbol}\n"]
    ranked = rank_patterns(results)

    if not ranked:
        lines.append("**Nenhum padrão passou nos critérios mínimos "
                      "(≥500 ocorrências, PF>1.30, p<0.05, estável 2024/2025/2026) nesta base de dados.**\n")
    else:
        lines.append("## Ranking Final (padrões aprovados)\n")
        for i, (name, r) in enumerate(ranked, 1):
            test_m = r["metrics"]["test"]
            val_m = r["metrics"]["val"]
            train_m = r["metrics"]["train"]
            lines.append(f"### {i}. {name}\n")
            for label, m in [("Treino 2024", train_m), ("Validação 2025", val_m), ("Teste 2026 (out-of-sample)", test_m)]:
                lines.append(
                    f"- **{label}**: n={m['n']}, win_rate={m['win_rate']:.1%} "
                    f"(IC95% {m['win_rate_ci95'][0]:.1%}–{m['win_rate_ci95'][1]:.1%}), "
                    f"PF={m['profit_factor']:.2f}, expectância={m['expectancy_R']:.3f}R, "
                    f"p={m['p_value_bootstrap']:.4f}, MAE={m['mae_mean']:.5f}, MFE={m['mfe_mean']:.5f}, "
                    f"barras até desfecho={m['bars_to_outcome_mean']:.1f}"
                )
            lines.append("")

    lines.append("\n## Padrões reprovados e motivo\n")
    for name, r in results.items():
        if not r["approved"]:
            lines.append(f"- **{name}**: {'; '.join(r['reasons'])}")

    return "\n".join(lines)
