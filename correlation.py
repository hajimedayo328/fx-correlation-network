"""通貨間相関行列の計算.

終値DataFrameからリターンを計算し、ピアソン相関行列を生成する。
"""

from __future__ import annotations

import pandas as pd


def calc_correlation_matrix(
    closes: pd.DataFrame,
    method: str = "pearson",
) -> pd.DataFrame:
    """終値DataFrameから相関行列を計算する.

    Args:
        closes: index=time, columns=通貨ペア名 の終値DataFrame
        method: 相関計算手法 ("pearson", "spearman", "kendall")

    Returns:
        通貨ペア間の相関行列
    """
    returns = closes.pct_change().dropna()

    if len(returns) < 10:
        raise ValueError(f"データ不足: {len(returns)}行（最低10行必要）")

    return returns.corr(method=method)


def filter_by_threshold(
    corr_matrix: pd.DataFrame,
    threshold: float = 0.3,
) -> pd.DataFrame:
    """閾値以下の相関をゼロにする.

    Args:
        corr_matrix: 相関行列
        threshold: 相関の絶対値がこれ以下のものを0にする

    Returns:
        フィルター済み相関行列
    """
    filtered = corr_matrix.copy()
    filtered[filtered.abs() < threshold] = 0.0
    return filtered


def get_edge_list(
    corr_matrix: pd.DataFrame,
    threshold: float = 0.3,
) -> list[dict]:
    """相関行列からエッジリスト（ペアと相関値）を生成する.

    Args:
        corr_matrix: 相関行列
        threshold: この絶対値以上の相関のみ含める

    Returns:
        [{"source": "EURUSD", "target": "GBPUSD", "weight": 0.85}, ...]
    """
    edges = []
    symbols = corr_matrix.columns.tolist()

    for i, src in enumerate(symbols):
        for j, tgt in enumerate(symbols):
            if i >= j:
                continue
            corr = corr_matrix.loc[src, tgt]
            if abs(corr) >= threshold:
                edges.append({
                    "source": src,
                    "target": tgt,
                    "weight": round(corr, 4),
                })

    return edges
