"""
Motor de mineração de padrões estruturais de topos/fundos.

Cada função `detect_*` varre a lista de swings (ver market_structure.py) e
retorna uma lista de "ocorrências": dicts com entry_time, entry_price,
stop_price, target_price (nível natural de retorno) e metadados. As
métricas de RR/expectância/etc são calculadas depois em metrics.py, andando
para frente nos candles reais (MAE/MFE) para não usar look-ahead.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .market_structure import Swing


def _fwd_outcome(df: pd.DataFrame, start_idx: int, direction: str, stop: float, target: float, max_bars: int = 500):
    """Caminha para frente candle a candle a partir de start_idx e descobre
    se stop ou target foi atingido primeiro (sem look-ahead), além de MAE/MFE."""
    n = len(df)
    end = min(n, start_idx + max_bars)
    if start_idx >= n:
        return None
    entry = df["close"].iloc[start_idx]
    mae = 0.0
    mfe = 0.0
    for i in range(start_idx + 1, end):
        hi = df["high"].iloc[i]
        lo = df["low"].iloc[i]
        if direction == "long":
            mfe = max(mfe, hi - entry)
            mae = max(mae, entry - lo)
            hit_target = hi >= target
            hit_stop = lo <= stop
        else:
            mfe = max(mfe, entry - lo)
            mae = max(mae, hi - entry)
            hit_target = lo <= target
            hit_stop = hi >= stop

        if hit_target and hit_stop:
            # ambos na mesma vela: assume o pior caso (stop primeiro) — conservador
            return {"result": "stop", "bars": i - start_idx, "mae": mae, "mfe": mfe}
        if hit_target:
            return {"result": "target", "bars": i - start_idx, "mae": mae, "mfe": mfe}
        if hit_stop:
            return {"result": "stop", "bars": i - start_idx, "mae": mae, "mfe": mfe}

    return {"result": "timeout", "bars": end - start_idx - 1, "mae": mae, "mfe": mfe}


def detect_ascending_bottoms_breakout(df: pd.DataFrame, swings: list[Swing], n_bottoms: int = 3, tol: float = 0.0):
    """Ex.1 do prompt: após N fundos ascendentes, o rompimento seguinte
    tende a retornar ao topo anterior. Alvo = topo entre os fundos; stop =
    último fundo (ou o fundo mais baixo da sequência)."""
    lows = [s for s in swings if s.kind == "low"]
    highs = {s.idx: s for s in swings if s.kind == "high"}
    occurrences = []

    for i in range(len(lows) - n_bottoms):
        seq = lows[i:i + n_bottoms]
        if all(seq[j + 1].price > seq[j].price * (1 + tol) for j in range(len(seq) - 1)):
            # topo de referência: maior swing high entre o primeiro e o último fundo da sequência
            between_highs = [s for s in swings if s.kind == "high" and seq[0].idx < s.idx < seq[-1].idx]
            if not between_highs:
                continue
            ref_high = max(between_highs, key=lambda s: s.price)
            entry_idx = seq[-1].idx
            entry_price = seq[-1].price
            stop_price = min(s.price for s in seq)
            target_price = ref_high.price
            occurrences.append({
                "pattern": "ascending_bottoms_breakout",
                "n_bottoms": n_bottoms,
                "entry_time": seq[-1].time, "entry_idx": entry_idx,
                "direction": "long",
                "entry_price": entry_price, "stop_price": stop_price, "target_price": target_price,
            })
    return occurrences


def detect_double_top_bottom(df: pd.DataFrame, swings: list[Swing], tol: float = 0.0015, kind: str = "top"):
    """Topo duplo / fundo duplo: dois swings do mesmo tipo com preços próximos
    (dentro de `tol` relativo), separados por um swing oposto (o "neckline").
    Alvo = projeção simétrica da distância até o neckline (medição clássica)."""
    target_kind = "high" if kind == "top" else "low"
    seq = [s for s in swings if s.kind in (target_kind,)]
    opp = {s.idx: s for s in swings if s.kind != target_kind}
    occurrences = []

    for a, b in zip(seq[:-1], seq[1:]):
        if abs(b.price - a.price) / a.price > tol:
            continue
        between = [s for s in swings if a.idx < s.idx < b.idx and s.kind != target_kind]
        if not between:
            continue
        neckline = min(between, key=lambda s: s.idx)
        measured_move = abs(a.price - neckline.price)
        if kind == "top":
            target = neckline.price - measured_move
            stop = max(a.price, b.price)
            direction = "short"
        else:
            target = neckline.price + measured_move
            stop = min(a.price, b.price)
            direction = "long"

        occurrences.append({
            "pattern": f"double_{kind}",
            "entry_time": b.time, "entry_idx": b.idx,
            "direction": direction,
            "entry_price": neckline.price, "stop_price": stop, "target_price": target,
            "symmetry_ratio": abs(b.price - a.price) / a.price,
        })
    return occurrences


def detect_liquidity_sweep_rejection(df: pd.DataFrame, swings: list[Swing], confirm_bars: int = 3, rr_target: float = 1.0):
    """Ex.3 do prompt: rompimento falso de um swing seguido de vela de
    rejeição -> alvo natural em 1R (parametrizável) medido a partir do
    stop (além do nível varrido) até a entrada na rejeição."""
    occurrences = []
    n = len(df)
    for s in swings:
        if s.idx + confirm_bars + 1 >= n:
            continue
        window = df.iloc[s.idx + 1: s.idx + 1 + confirm_bars]
        if s.kind == "high":
            swept = (window["high"] > s.price)
            if not swept.any():
                continue
            sweep_pos = window.index[swept][0]
            sweep_i = df.index.get_loc(sweep_pos)
            rejection_close = df["close"].iloc[sweep_i]
            if rejection_close >= s.price:
                continue  # não rejeitou
            entry = rejection_close
            stop = df["high"].iloc[sweep_i]
            risk = stop - entry
            target = entry - risk * rr_target
            direction = "short"
        else:
            swept = (window["low"] < s.price)
            if not swept.any():
                continue
            sweep_pos = window.index[swept][0]
            sweep_i = df.index.get_loc(sweep_pos)
            rejection_close = df["close"].iloc[sweep_i]
            if rejection_close <= s.price:
                continue
            entry = rejection_close
            stop = df["low"].iloc[sweep_i]
            risk = entry - stop
            target = entry + risk * rr_target
            direction = "long"

        if risk <= 0:
            continue
        occurrences.append({
            "pattern": "liquidity_sweep_rejection",
            "entry_time": df.index[sweep_i], "entry_idx": sweep_i,
            "direction": direction,
            "entry_price": entry, "stop_price": stop, "target_price": target,
            "swept_level": s.price,
        })
    return occurrences


def detect_movement_symmetry(df: pd.DataFrame, swings: list[Swing], tol: float = 0.15):
    """Procura pares de pernas consecutivas (swing->swing) cuja amplitude A e
    amplitude B (a seguinte) fiquem dentro de `tol` relativo uma da outra —
    mede simetria estrutural (perna de impulso ~ perna de correção/expansão)."""
    occurrences = []
    for i in range(len(swings) - 2):
        a0, a1, a2 = swings[i], swings[i + 1], swings[i + 2]
        leg_a = abs(a1.price - a0.price)
        leg_b = abs(a2.price - a1.price)
        if leg_a == 0:
            continue
        ratio = leg_b / leg_a
        if abs(ratio - 1.0) <= tol or abs(ratio - 0.5) <= tol or abs(ratio - 2.0) <= tol:
            direction = "long" if a2.kind == "high" else "short"
            entry = a2.price
            stop = a1.price
            risk = abs(entry - stop)
            target = entry + risk if direction == "long" else entry - risk
            occurrences.append({
                "pattern": "movement_symmetry",
                "entry_time": a2.time, "entry_idx": a2.idx,
                "direction": direction,
                "entry_price": entry, "stop_price": stop, "target_price": target,
                "leg_a": leg_a, "leg_b": leg_b, "ratio": ratio,
            })
    return occurrences


PATTERN_REGISTRY = {
    "ascending_bottoms_breakout": detect_ascending_bottoms_breakout,
    "double_top": lambda df, sw: detect_double_top_bottom(df, sw, kind="top"),
    "double_bottom": lambda df, sw: detect_double_top_bottom(df, sw, kind="bottom"),
    "liquidity_sweep_rejection": detect_liquidity_sweep_rejection,
    "movement_symmetry": detect_movement_symmetry,
}


def attach_outcomes(df: pd.DataFrame, occurrences: list[dict], max_bars: int = 500) -> pd.DataFrame:
    """Para cada ocorrência, caminha os candles reais à frente (sem
    look-ahead) e anexa resultado (target/stop/timeout), MAE, MFE e RR."""
    rows = []
    for occ in occurrences:
        outcome = _fwd_outcome(
            df, occ["entry_idx"], occ["direction"],
            occ["stop_price"], occ["target_price"], max_bars=max_bars,
        )
        if outcome is None:
            continue
        risk = abs(occ["entry_price"] - occ["stop_price"])
        reward = abs(occ["target_price"] - occ["entry_price"])
        row = dict(occ)
        row.update(outcome)
        row["risk"] = risk
        row["reward"] = reward
        row["rr_planned"] = reward / risk if risk > 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)
