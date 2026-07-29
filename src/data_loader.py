"""
Carregamento de dados OHLC (ou ticks) exportados do MT5.

Formatos aceitos (auto-detectados pelo cabeçalho):
- MT5 CSV padrão:  <DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<TICKVOL>,<VOL>,<SPREAD>
- Genérico:         datetime,open,high,low,close,volume
- Ticks:            datetime,bid,ask  (ou date,time,bid,ask,last,volume)

Nenhum indicador é calculado aqui. Apenas OHLC, tempo e volume/tick.
"""
from __future__ import annotations

import pandas as pd
import numpy as np


REQUIRED_COLS = ["open", "high", "low", "close"]


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower().replace("<", "").replace(">", "") for c in df.columns]
    rename_map = {
        "tickvol": "volume",
        "vol": "volume",
        "last": "close",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    return df


def load_ohlc_csv(path: str, symbol: str | None = None) -> pd.DataFrame:
    """Carrega um CSV de candles e retorna DataFrame indexado por datetime,
    ordenado, sem duplicatas, com colunas: open, high, low, close, volume, symbol, year.
    """
    raw = pd.read_csv(path, sep=None, engine="python")
    df = _normalize_columns(raw)

    if "date" in df.columns and "time" in df.columns:
        dt = pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str), errors="coerce")
    elif "datetime" in df.columns:
        dt = pd.to_datetime(df["datetime"], errors="coerce")
    elif "date" in df.columns:
        dt = pd.to_datetime(df["date"], errors="coerce")
    else:
        raise ValueError(f"Não foi possível identificar coluna de data/hora em {path}. Colunas: {list(df.columns)}")

    df["datetime"] = dt

    # Ticks: bid/ask -> sintetiza "close" para compatibilidade com o motor de candles.
    if "bid" in df.columns and "open" not in df.columns:
        mid = df["bid"] if "ask" not in df.columns else (df["bid"] + df["ask"]) / 2.0
        df["open"] = df["high"] = df["low"] = df["close"] = mid

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Colunas ausentes {missing} em {path}. Colunas encontradas: {list(df.columns)}")

    if "volume" not in df.columns:
        df["volume"] = np.nan

    df = df.dropna(subset=["datetime", "open", "high", "low", "close"])
    df = df.sort_values("datetime").drop_duplicates(subset="datetime")
    df = df.set_index("datetime")
    df["year"] = df.index.year
    df["symbol"] = symbol or "UNKNOWN"

    return df[["symbol", "open", "high", "low", "close", "volume", "year"]]


def ticks_to_candles(path: str, timeframe: str = "1min", symbol: str | None = None) -> pd.DataFrame:
    """Converte um arquivo de ticks em candles no menor timeframe desejado,
    usando somente preço (bid/mid) — nenhum indicador."""
    raw = pd.read_csv(path, sep=None, engine="python")
    df = _normalize_columns(raw)

    if "date" in df.columns and "time" in df.columns:
        dt = pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str), errors="coerce")
    else:
        dt = pd.to_datetime(df["datetime"], errors="coerce")

    price = df["bid"] if "ask" not in df.columns else (df["bid"] + df["ask"]) / 2.0
    s = pd.Series(price.values, index=dt).dropna().sort_index()

    ohlc = s.resample(timeframe).ohlc()
    ohlc["volume"] = s.resample(timeframe).count()
    ohlc = ohlc.dropna(subset=["open", "high", "low", "close"])
    ohlc["year"] = ohlc.index.year
    ohlc["symbol"] = symbol or "UNKNOWN"
    return ohlc[["symbol", "open", "high", "low", "close", "volume", "year"]]


def split_by_year(df: pd.DataFrame, train_year=2024, val_year=2025, test_year=2026):
    """Divide em treino/validação/teste por ano, conforme exigido para evitar overfitting."""
    return {
        "train": df[df["year"] == train_year].copy(),
        "val": df[df["year"] == val_year].copy(),
        "test": df[df["year"] == test_year].copy(),
    }
