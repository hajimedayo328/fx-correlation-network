"""経済指標波及ネットワーク（リードラグ分析）.

通貨ペア間の「先行・遅行関係」をクロス相関で検出し、
情報がどの通貨からどの通貨に波及するかを有向グラフで可視化。

例: 米雇用統計発表 → USDJPYが最初に動く → EURJPY/GBPJPYが追随

グラフ理論的意味:
- 有向グラフ: A→Bは「Aの値動きがBに先行する」
- エッジの重み = クロス相関の強さ
- 入次数が高い = 他のペアに追随する（遅行指標）
- 出次数が高い = 他のペアを先導する（先行指標）
- Granger因果性のグラフ版
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import networkx as nx


def calc_cross_correlation(
    series_a: pd.Series,
    series_b: pd.Series,
    max_lag: int = 12,
) -> tuple[int, float]:
    """2つの時系列間のクロス相関を計算し、最適ラグを返す.

    正のラグ = series_aがseries_bに先行
    負のラグ = series_bがseries_aに先行

    Args:
        series_a: 時系列A
        series_b: 時系列B
        max_lag: 検査するラグの最大値

    Returns:
        (最適ラグ, そのラグでの相関係数)
    """
    a = series_a.dropna().values
    b = series_b.dropna().values
    n = min(len(a), len(b))

    if n < max_lag * 2:
        return 0, 0.0

    a = a[:n]
    b = b[:n]

    # 正規化
    a_norm = (a - np.mean(a)) / (np.std(a) + 1e-10)
    b_norm = (b - np.mean(b)) / (np.std(b) + 1e-10)

    best_lag = 0
    best_corr = 0.0

    for lag in range(-max_lag, max_lag + 1):
        if lag == 0:
            continue
        if lag > 0:
            corr = np.mean(a_norm[:-lag] * b_norm[lag:])
        else:
            corr = np.mean(a_norm[-lag:] * b_norm[:lag])

        if abs(corr) > abs(best_corr):
            best_corr = corr
            best_lag = lag

    return best_lag, round(best_corr, 4)


def build_lead_lag_network(
    closes: pd.DataFrame,
    max_lag: int = 12,
    min_correlation: float = 0.15,
    use_returns: bool = True,
) -> nx.DiGraph:
    """リードラグネットワークを構築する.

    Args:
        closes: 終値DataFrame（columns=通貨ペア）
        max_lag: 検査するラグの最大値（バー数）
        min_correlation: エッジを追加する最低相関
        use_returns: Trueならリターン、Falseなら終値でクロス相関

    Returns:
        有向グラフ（A→B = AがBに先行）
    """
    if use_returns:
        data = closes.pct_change().dropna()
    else:
        data = closes.dropna()

    symbols = data.columns.tolist()
    G = nx.DiGraph()
    G.add_nodes_from(symbols)

    for i, src in enumerate(symbols):
        for j, tgt in enumerate(symbols):
            if i >= j:
                continue

            lag, corr = calc_cross_correlation(
                data[src], data[tgt], max_lag=max_lag
            )

            if abs(corr) < min_correlation:
                continue

            # 正のラグ = srcがtgtに先行
            if lag > 0:
                G.add_edge(src, tgt, lag=lag, correlation=corr, weight=abs(corr))
            elif lag < 0:
                G.add_edge(tgt, src, lag=-lag, correlation=corr, weight=abs(corr))

    return G


def calc_lead_lag_matrix(
    closes: pd.DataFrame,
    max_lag: int = 12,
    use_returns: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """全ペア間のリードラグ行列を計算する.

    Returns:
        (ラグ行列, 相関行列)
        lag_matrix[A][B] > 0 → AがBに先行
    """
    if use_returns:
        data = closes.pct_change().dropna()
    else:
        data = closes.dropna()

    symbols = data.columns.tolist()
    n = len(symbols)

    lag_matrix = pd.DataFrame(0, index=symbols, columns=symbols, dtype=int)
    corr_matrix = pd.DataFrame(0.0, index=symbols, columns=symbols)

    for i in range(n):
        for j in range(i + 1, n):
            lag, corr = calc_cross_correlation(
                data[symbols[i]], data[symbols[j]], max_lag=max_lag
            )
            lag_matrix.iloc[i, j] = lag
            lag_matrix.iloc[j, i] = -lag
            corr_matrix.iloc[i, j] = corr
            corr_matrix.iloc[j, i] = corr

    return lag_matrix, corr_matrix


def calc_lead_lag_metrics(G: nx.DiGraph) -> pd.DataFrame:
    """リードラグネットワークの指標を計算する."""
    if len(G.nodes()) == 0:
        return pd.DataFrame()

    in_deg = dict(G.in_degree(weight="weight"))
    out_deg = dict(G.out_degree(weight="weight"))

    rows = []
    for node in G.nodes():
        in_d = in_deg.get(node, 0)
        out_d = out_deg.get(node, 0)

        # 先行スコア = 出次数 - 入次数（正=先行、負=遅行）
        lead_score = out_d - in_d

        # 平均先行ラグ
        out_edges = [(v, d["lag"]) for _, v, d in G.out_edges(node, data=True)]
        avg_lag = np.mean([lag for _, lag in out_edges]) if out_edges else 0

        rows.append({
            "通貨ペア": node,
            "先行スコア": round(lead_score, 3),
            "出次数（先導）": round(out_d, 3),
            "入次数（追随）": round(in_d, 3),
            "平均先行ラグ": round(avg_lag, 1),
            "先導先": ", ".join(v for v, _ in out_edges[:3]) if out_edges else "-",
        })

    return (
        pd.DataFrame(rows)
        .sort_values("先行スコア", ascending=False)
        .reset_index(drop=True)
    )


def detect_propagation_events(
    closes: pd.DataFrame,
    window: int = 5,
    vol_threshold: float = 2.0,
) -> pd.DataFrame:
    """高ボラティリティイベント（経済指標発表等）を検出し、
    伝播パターンを分析する.

    Args:
        closes: 終値DataFrame
        window: ボラティリティ計算ウィンドウ
        vol_threshold: 平均+何σで「イベント」と判定

    Returns:
        イベント一覧（時刻、最初に動いたペア、伝播順序）
    """
    returns = closes.pct_change().dropna()
    abs_returns = returns.abs()

    # 各ペアのボラティリティスパイクを検出
    mean_vol = abs_returns.rolling(50).mean()
    std_vol = abs_returns.rolling(50).std()
    threshold = mean_vol + vol_threshold * std_vol

    events = []
    checked_times = set()

    for t in returns.index:
        if t in checked_times:
            continue

        # この時刻でスパイクしたペアを検出
        spiked = []
        for pair in returns.columns:
            if t in threshold.index and abs_returns.loc[t, pair] > threshold.loc[t, pair]:
                spiked.append((pair, abs_returns.loc[t, pair]))

        if len(spiked) >= 2:  # 2ペア以上同時にスパイク = イベント
            spiked.sort(key=lambda x: x[1], reverse=True)
            events.append({
                "時刻": t,
                "影響ペア数": len(spiked),
                "最大反応ペア": spiked[0][0],
                "最大反応(%)": round(spiked[0][1] * 100, 3),
                "伝播順序": " → ".join(p for p, _ in spiked[:5]),
            })

            # 近接時刻をスキップ
            t_idx = returns.index.get_loc(t)
            for skip in range(1, window + 1):
                if t_idx + skip < len(returns.index):
                    checked_times.add(returns.index[t_idx + skip])

    if not events:
        return pd.DataFrame(
            columns=["時刻", "影響ペア数", "最大反応ペア", "最大反応(%)", "伝播順序"]
        )

    return pd.DataFrame(events).sort_values("最大反応(%)", ascending=False).head(20)
