"""
Estrutura de mercado pura (sem indicadores): fractais, swing highs/lows,
BOS/CHoCH, rompimentos falsos e sweeps de liquidez.

Tudo derivado apenas de: open, high, low, close, tempo, sequência de velas.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class Swing:
    idx: int            # posição inteira na série
    time: pd.Timestamp
    price: float
    kind: str            # "high" ou "low"


def find_fractals(df: pd.DataFrame, left: int = 2, right: int = 2) -> pd.DataFrame:
    """Fractais de Williams generalizados: uma máxima é fractal se for a maior
    entre `left` velas antes e `right` velas depois (idem para mínima)."""
    highs = df["high"].values
    lows = df["low"].values
    n = len(df)
    is_high = np.zeros(n, dtype=bool)
    is_low = np.zeros(n, dtype=bool)

    for i in range(left, n - right):
        window_h = highs[i - left:i + right + 1]
        window_l = lows[i - left:i + right + 1]
        if highs[i] == window_h.max() and np.argmax(window_h) == left:
            is_high[i] = True
        if lows[i] == window_l.min() and np.argmin(window_l) == left:
            is_low[i] = True

    out = df.copy()
    out["fractal_high"] = is_high
    out["fractal_low"] = is_low
    return out


def extract_swings(df_fractal: pd.DataFrame) -> list[Swing]:
    """Converte fractais em uma sequência alternada de swing highs/lows
    (remove fractais consecutivos do mesmo tipo, mantendo o extremo mais forte)."""
    candidates = []
    for i, (t, row) in enumerate(df_fractal.iterrows()):
        if row["fractal_high"]:
            candidates.append(Swing(i, t, row["high"], "high"))
        if row["fractal_low"]:
            candidates.append(Swing(i, t, row["low"], "low"))
    candidates.sort(key=lambda s: s.idx)

    swings: list[Swing] = []
    for c in candidates:
        if not swings:
            swings.append(c)
            continue
        last = swings[-1]
        if c.kind == last.kind:
            # mantém o extremo mais forte (maior alta / menor baixa)
            if (c.kind == "high" and c.price > last.price) or (c.kind == "low" and c.price < last.price):
                swings[-1] = c
        else:
            swings.append(c)
    return swings


def detect_bos_choch(swings: list[Swing]) -> list[dict]:
    """Detecta Break of Structure (continuação) e Change of Character (reversão)
    comparando cada swing com o anterior do mesmo tipo."""
    events = []
    last_high = None
    last_low = None
    trend = None  # "up" | "down"

    for s in swings:
        if s.kind == "high":
            if last_high is not None:
                broke = s.price > last_high.price
                if broke:
                    kind = "BOS_up" if trend != "down" else "CHoCH_up"
                    events.append({"time": s.time, "idx": s.idx, "type": kind, "price": s.price, "ref_price": last_high.price})
                    trend = "up"
            last_high = s
        else:
            if last_low is not None:
                broke = s.price < last_low.price
                if broke:
                    kind = "BOS_down" if trend != "up" else "CHoCH_down"
                    events.append({"time": s.time, "idx": s.idx, "type": kind, "price": s.price, "ref_price": last_low.price})
                    trend = "down"
            last_low = s
    return events


def detect_false_breakouts(df: pd.DataFrame, swings: list[Swing], confirm_bars: int = 3) -> list[dict]:
    """Rompimento falso: preço ultrapassa um swing high/low e fecha de volta
    dentro do range dentro de `confirm_bars` velas (sem indicadores, só close vs nível)."""
    events = []
    n = len(df)
    for s in swings:
        if s.idx + confirm_bars >= n:
            continue
        window = df.iloc[s.idx + 1: s.idx + 1 + confirm_bars]
        if s.kind == "high":
            broke = (window["high"] > s.price).any()
            rejected = broke and (window["close"] < s.price).any()
        else:
            broke = (window["low"] < s.price).any()
            rejected = broke and (window["close"] > s.price).any()
        if broke and rejected:
            reject_pos = window.index[(window["close"] < s.price)] if s.kind == "high" else window.index[(window["close"] > s.price)]
            events.append({
                "time": s.time, "idx": s.idx, "level": s.price, "kind": s.kind,
                "rejection_time": reject_pos[0] if len(reject_pos) else None,
            })
    return events


def swing_amplitudes(swings: list[Swing]) -> pd.DataFrame:
    """Amplitude (distância em preço), duração (em candles) e velocidade
    (amplitude/candles) entre swings consecutivos."""
    rows = []
    for a, b in zip(swings[:-1], swings[1:]):
        amp = abs(b.price - a.price)
        candles = b.idx - a.idx
        rows.append({
            "start_time": a.time, "end_time": b.time,
            "start_kind": a.kind, "end_kind": b.kind,
            "start_price": a.price, "end_price": b.price,
            "amplitude": amp, "candles": candles,
            "speed": amp / candles if candles > 0 else np.nan,
        })
    return pd.DataFrame(rows)
