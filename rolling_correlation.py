"""ローリング相関分析.

過去データに対してウィンドウをスライドさせ、相関の時間変化を追跡する。
相関崩壊の検出やレジーム分類に使用。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def calc_rolling_correlation(
    closes: pd.DataFrame,
    window: int = 50,
) -> dict[str, pd.DataFrame]:
    """全ペア間のローリング相関を計算する.

    Args:
        closes: index=time, columns=通貨ペア名 の終値DataFrame
        window: ローリングウィンドウの足数

    Returns:
        {"EURUSD_GBPUSD": pd.Series(時刻→相関値), ...} の辞書
    """
    returns = closes.pct_change().dropna()
    symbols = returns.columns.tolist()
    rolling_corrs = {}

    for i, src in enumerate(symbols):
        for j, tgt in enumerate(symbols):
            if i >= j:
                continue
            key = f"{src} / {tgt}"
            rolling_corrs[key] = returns[src].rolling(window).corr(returns[tgt])

    return rolling_corrs


def calc_rolling_graph_density(
    closes: pd.DataFrame,
    window: int = 50,
    threshold: float = 0.3,
) -> pd.Series:
    """ローリングウィンドウでグラフ密度の時間変化を計算する.

    グラフ密度 = 閾値以上の相関ペア数 / 全ペア数
    高い = 市場全体が連動（トレンド相場の可能性）
    低い = バラバラに動く（レンジ or 転換期の可能性）

    Args:
        closes: 終値DataFrame
        window: ローリングウィンドウ
        threshold: 相関の閾値

    Returns:
        時刻→グラフ密度 のSeries
    """
    returns = closes.pct_change().dropna()
    symbols = returns.columns.tolist()
    n_pairs = len(symbols) * (len(symbols) - 1) / 2

    densities = []
    times = []

    for end in range(window, len(returns)):
        start = end - window
        window_returns = returns.iloc[start:end]
        corr = window_returns.corr()

        # 閾値以上のペア数をカウント（対角線除く上三角）
        count = 0
        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                if abs(corr.iloc[i, j]) >= threshold:
                    count += 1

        densities.append(count / n_pairs if n_pairs > 0 else 0)
        times.append(returns.index[end])

    return pd.Series(densities, index=times, name="graph_density")


def detect_correlation_breakdowns(
    rolling_corrs: dict[str, pd.Series],
    drop_threshold: float = 0.3,
    window: int = 10,
) -> pd.DataFrame:
    """相関の急変（崩壊）を検出する.

    ローリング相関が短期間で大きく変化したポイントを検出。

    Args:
        rolling_corrs: calc_rolling_correlationの返り値
        drop_threshold: この値以上の相関変化を「崩壊」とみなす
        window: 変化を測る期間

    Returns:
        時刻、ペア名、変化量のDataFrame
    """
    breakdowns = []

    for pair_name, corr_series in rolling_corrs.items():
        corr_clean = corr_series.dropna()
        if len(corr_clean) < window:
            continue

        corr_change = corr_clean.diff(window).abs()

        for time_idx, change in corr_change.items():
            if change >= drop_threshold:
                breakdowns.append({
                    "time": time_idx,
                    "pair": pair_name,
                    "change": round(change, 4),
                    "before": round(corr_clean.loc[:time_idx].iloc[-window - 1], 4)
                    if len(corr_clean.loc[:time_idx]) > window
                    else None,
                    "after": round(corr_clean.loc[time_idx], 4),
                })

    if not breakdowns:
        return pd.DataFrame(columns=["time", "pair", "change", "before", "after"])

    return (
        pd.DataFrame(breakdowns)
        .sort_values("time")
        .reset_index(drop=True)
    )
