#!/usr/bin/env python3
"""
CLI: descoberta estatística de padrões de topos/fundos (2024 treino, 2025
validação, 2026 teste out-of-sample). Sem indicadores — só OHLC/estrutura.

Uso:
    python run_discovery.py --csv data/EURUSD_M1_2024_2026.csv --symbol EURUSD

O CSV deve conter candles (menor timeframe disponível) de 2024 a 2026,
exportados do MT5 (ex: <DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<TICKVOL>).
Se você só tem ticks, converta antes com src.data_loader.ticks_to_candles.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pipeline import run_discovery, format_report


def main():
    ap = argparse.ArgumentParser(description="Descoberta de padrões de topos/fundos")
    ap.add_argument("--csv", required=True, help="Caminho do CSV de candles (2024-2026)")
    ap.add_argument("--symbol", required=True, help="Nome do ativo (ex: EURUSD, WIN, WDO)")
    ap.add_argument("--fractal-left", type=int, default=2)
    ap.add_argument("--fractal-right", type=int, default=2)
    ap.add_argument("--min-occurrences", type=int, default=500)
    ap.add_argument("--min-profit-factor", type=float, default=1.30)
    ap.add_argument("--max-p-value", type=float, default=0.05)
    ap.add_argument("--out", default=None, help="Arquivo de saída (markdown). Default: reports/<symbol>.md")
    args = ap.parse_args()

    results = run_discovery(
        args.csv, args.symbol,
        fractal_left=args.fractal_left, fractal_right=args.fractal_right,
        min_occurrences=args.min_occurrences,
        min_profit_factor=args.min_profit_factor,
        max_p_value=args.max_p_value,
    )
    report = format_report(results, args.symbol)

    out_path = args.out or os.path.join("reports", f"{args.symbol}.md")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(report)

    print(report)
    print(f"\nRelatório salvo em: {out_path}")


if __name__ == "__main__":
    main()
