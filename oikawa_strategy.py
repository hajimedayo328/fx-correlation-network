"""及川式通貨相関トレード戦略.

及川圭哉氏の手法をベースにした通貨相関 + ピボット + MA方向フィルター戦略。

核心ロジック:
1. 10通貨ペアから通貨強弱を判定（EUR/GBP/USD/JPY/AUD）
2. 最強通貨 × 最弱通貨のペアを選択
3. デイリーピボットP0への接触で反発エントリー
4. 上位足MA方向でフィルター（トレンドに逆らわない）

参考: 及川圭哉「ガチ速FX」/ FXism
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# 及川式10通貨ペア
OIKAWA_PAIRS = [
    "USDJPY", "EURJPY", "GBPJPY", "AUDJPY",
    "EURUSD", "GBPUSD", "AUDUSD",
    "EURGBP", "EURAUD", "GBPAUD",
]

# 各通貨ペアの構成通貨 (base, quote)
PAIR_CURRENCIES = {
    "USDJPY": ("USD", "JPY"),
    "EURJPY": ("EUR", "JPY"),
    "GBPJPY": ("GBP", "JPY"),
    "AUDJPY": ("AUD", "JPY"),
    "EURUSD": ("EUR", "USD"),
    "GBPUSD": ("GBP", "USD"),
    "AUDUSD": ("AUD", "USD"),
    "EURGBP": ("EUR", "GBP"),
    "EURAUD": ("EUR", "AUD"),
    "GBPAUD": ("GBP", "AUD"),
}


@dataclass(frozen=True)
class OikawaSignal:
    """及川式エントリーシグナル."""

    time: pd.Timestamp
    pair: str
    direction: str  # "long" or "short"
    entry_price: float
    sl_price: float
    tp_price: float
    pivot_p: float
    strongest: str  # 最強通貨
    weakest: str  # 最弱通貨
    strength_score: float  # 強弱スコア差


def calc_daily_pivots(daily_df: pd.DataFrame) -> pd.DataFrame:
    """日足データからデイリーピボットを計算する.

    Args:
        daily_df: columns=[open, high, low, close], index=日付

    Returns:
        P, R1, R2, R3, S1, S2, S3 を含むDataFrame
    """
    h = daily_df["high"]
    l = daily_df["low"]
    c = daily_df["close"]

    p = (h + l + c) / 3
    r1 = 2 * p - l
    r2 = p + (h - l)
    r3 = h + 2 * (p - l)
    s1 = 2 * p - h
    s2 = p - (h - l)
    s3 = l - 2 * (h - p)

    return pd.DataFrame({
        "P": p, "R1": r1, "R2": r2, "R3": r3,
        "S1": s1, "S2": s2, "S3": s3,
    }, index=daily_df.index)


def calc_currency_strength(
    closes_dict: dict[str, pd.DataFrame],
    lookback: int = 20,
) -> pd.DataFrame:
    """通貨強弱を計算する.

    各通貨ペアのリターンから、個別通貨の強弱スコアを算出。
    及川式: 複数ペアの方向を比較して最強・最弱を特定。

    Args:
        closes_dict: {ペア名: DataFrame(close列含む)} の辞書
        lookback: 強弱計算の期間

    Returns:
        columns=通貨名, index=時刻 のスコアDataFrame
    """
    currencies = ["EUR", "GBP", "USD", "JPY", "AUD"]

    # 各ペアのリターンを計算
    returns = {}
    for pair, df in closes_dict.items():
        if "close" in df.columns:
            returns[pair] = df["close"].pct_change(lookback)
        elif len(df.columns) == 1:
            returns[pair] = df.iloc[:, 0].pct_change(lookback)

    # 通貨別スコア: base通貨が上がる=baseが強い、quote通貨が上がる=quoteが弱い
    # 共通のインデックスを取得
    common_idx = None
    for pair, ret in returns.items():
        if pair not in PAIR_CURRENCIES:
            continue
        idx = ret.dropna().index
        if common_idx is None:
            common_idx = idx
        else:
            common_idx = common_idx.intersection(idx)

    if common_idx is None or len(common_idx) == 0:
        return pd.DataFrame()

    strength = pd.DataFrame(0.0, index=common_idx, columns=currencies)

    for pair, ret in returns.items():
        if pair not in PAIR_CURRENCIES:
            continue
        base, quote = PAIR_CURRENCIES[pair]
        aligned = ret.reindex(common_idx).fillna(0)
        if base in currencies:
            strength[base] += aligned
        if quote in currencies:
            strength[quote] -= aligned

    # 通貨数で正規化
    for cur in currencies:
        n_pairs = sum(
            1 for p, (b, q) in PAIR_CURRENCIES.items()
            if (b == cur or q == cur) and p in returns
        )
        if n_pairs > 0:
            strength[cur] /= n_pairs

    return strength


def find_strongest_weakest(
    strength_row: pd.Series,
) -> tuple[str, str, float]:
    """最強通貨と最弱通貨を特定する.

    Returns:
        (最強通貨, 最弱通貨, スコア差)
    """
    strongest = strength_row.idxmax()
    weakest = strength_row.idxmin()
    score_diff = strength_row[strongest] - strength_row[weakest]
    return strongest, weakest, score_diff


def get_pair_for_currencies(strong: str, weak: str) -> str | None:
    """最強×最弱通貨からトレードペアと方向を決定する.

    Returns:
        ペア名（存在しない組み合わせはNone）
    """
    # strong/weak → ペア名を探す
    for pair, (base, quote) in PAIR_CURRENCIES.items():
        if base == strong and quote == weak:
            return pair
        if base == weak and quote == strong:
            return pair
    return None


def get_direction_for_pair(
    pair: str, strong: str, weak: str,
) -> str:
    """ペアに対するトレード方向を決定する."""
    base, quote = PAIR_CURRENCIES[pair]
    if base == strong:
        return "long"  # base通貨が強い → 買い
    return "short"  # base通貨が弱い → 売り


def is_near_pivot(
    price: float,
    pivot_p: float,
    tolerance_pips: float = 15.0,
    pair: str = "USDJPY",
) -> bool:
    """価格がピボットP0付近にあるか判定する.

    Args:
        price: 現在価格
        pivot_p: ピボットP0
        tolerance_pips: 許容範囲(pips)
        pair: 通貨ペア（pip値の計算用）
    """
    pip_value = _get_pip_value(pair)
    distance = abs(price - pivot_p) / pip_value
    return distance <= tolerance_pips


def _get_pip_value(pair: str) -> float:
    """通貨ペアの1pip値を返す."""
    if "JPY" in pair:
        return 0.01
    return 0.0001


def check_ma_direction(
    closes: pd.Series,
    ma_period: int = 120,
    direction: str = "long",
) -> bool:
    """MA方向フィルター.

    5分足120MA = 1時間足10MA相当。
    MAが上向き → ロングのみ許可
    MAが下向き → ショートのみ許可

    Args:
        closes: 終値Series
        ma_period: MA期間
        direction: チェックしたい方向

    Returns:
        MA方向とdirectionが一致すればTrue
    """
    if len(closes) < ma_period + 5:
        return False

    ma = closes.rolling(ma_period).mean()
    ma_current = ma.iloc[-1]
    ma_prev = ma.iloc[-6]  # 5本前と比較

    if np.isnan(ma_current) or np.isnan(ma_prev):
        return False

    ma_rising = ma_current > ma_prev
    if direction == "long":
        return ma_rising
    return not ma_rising


def calc_sl_tp(
    entry_price: float,
    direction: str,
    pivot_p: float,
    pivot_s1: float,
    pivot_r1: float,
    pair: str,
    sl_buffer_pips: float = 5.0,
) -> tuple[float, float]:
    """SL/TPを計算する.

    SL: ピボットの反対側 + バッファ
    TP: 次のピボットレベル

    Args:
        entry_price: エントリー価格
        direction: "long" or "short"
        pivot_p: ピボットP0
        pivot_s1: S1
        pivot_r1: R1
        pair: 通貨ペア
        sl_buffer_pips: SLバッファ(pips)

    Returns:
        (sl_price, tp_price)
    """
    pip_value = _get_pip_value(pair)
    buffer = sl_buffer_pips * pip_value

    if direction == "long":
        sl = pivot_s1 - buffer
        tp = pivot_r1
    else:
        sl = pivot_r1 + buffer
        tp = pivot_s1

    return sl, tp
