"""オーダーフローネットワーク分析.

ティックボリュームの急増パターンから売買圧の伝播構造を可視化する。
「大口が動いた通貨」→「連鎖して動いた通貨」の方向付きネットワークを構築。

MT5のティックボリュームを使用（実際の注文板データはFXでは制限があるため、
ボリュームスパイクを大口注文の代理指標として使用）。

グラフ理論的意味:
- 有向グラフ（DiGraph）: A→Bは「Aのボリュームスパイク後にBが動く」
- 重み = 伝播の頻度・強さ
- PageRank = 情報の発信源としての重要度
- 入次数/出次数 = 「影響を受けやすい」/「影響を与えやすい」
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import networkx as nx


def detect_volume_spikes(
    volumes: pd.Series,
    threshold_std: float = 2.0,
    min_lookback: int = 20,
) -> pd.Series:
    """ティックボリュームのスパイク（異常増加）を検出する.

    Args:
        volumes: ティックボリュームのSeries
        threshold_std: 平均+何σでスパイク判定
        min_lookback: ローリング平均の最小期間

    Returns:
        スパイクフラグ（True/False）のSeries
    """
    rolling_mean = volumes.rolling(min_lookback).mean()
    rolling_std = volumes.rolling(min_lookback).std()

    spike_threshold = rolling_mean + threshold_std * rolling_std
    return volumes > spike_threshold


def calc_volume_flow_network(
    ohlcv_dict: dict[str, pd.DataFrame],
    spike_threshold: float = 2.0,
    lag_bars: int = 3,
    min_response_pct: float = 0.1,
) -> nx.DiGraph:
    """ボリュームスパイクの伝播からオーダーフローネットワークを構築する.

    ロジック:
    1. 各ペアのボリュームスパイクを検出
    2. ペアAでスパイク発生後、lag_bars以内にペアBで大きな値動きがあれば
       A→Bのエッジを追加
    3. 伝播の頻度と強さを重みとする

    Args:
        ohlcv_dict: {ペア名: DataFrame(open,high,low,close,volume)}
        spike_threshold: スパイク判定のσ倍数
        lag_bars: 伝播を検出するラグ（バー数）
        min_response_pct: 最小反応率（%）

    Returns:
        有向グラフ（DiGraph）
    """
    pairs = list(ohlcv_dict.keys())

    # 各ペアのスパイクと価格変化を計算
    spikes = {}
    returns = {}
    for pair, df in ohlcv_dict.items():
        if "volume" in df.columns and "close" in df.columns:
            spikes[pair] = detect_volume_spikes(
                df["volume"], threshold_std=spike_threshold
            )
            returns[pair] = df["close"].pct_change().abs() * 100  # %表記

    # 共通インデックス
    common_idx = None
    for pair in spikes:
        idx = spikes[pair].dropna().index
        if common_idx is None:
            common_idx = idx
        else:
            common_idx = common_idx.intersection(idx)

    if common_idx is None or len(common_idx) < 20:
        return nx.DiGraph()

    # 伝播検出
    flow_counts: dict[tuple[str, str], int] = {}
    flow_strengths: dict[tuple[str, str], list[float]] = {}

    for src in spikes:
        src_spikes = spikes[src].reindex(common_idx).fillna(False)
        spike_times = common_idx[src_spikes]

        for tgt in spikes:
            if src == tgt:
                continue

            tgt_returns = returns[tgt].reindex(common_idx).fillna(0)
            tgt_mean = tgt_returns.mean()

            count = 0
            strengths = []

            for t in spike_times:
                # スパイク後lag_bars以内の値動きをチェック
                t_idx = common_idx.get_loc(t)
                end_idx = min(t_idx + lag_bars + 1, len(common_idx))

                if end_idx <= t_idx + 1:
                    continue

                response = tgt_returns.iloc[t_idx + 1:end_idx].max()
                if response > max(min_response_pct, tgt_mean * 1.5):
                    count += 1
                    strengths.append(response)

            if count > 0:
                flow_counts[(src, tgt)] = count
                flow_strengths[(src, tgt)] = strengths

    # グラフ構築
    G = nx.DiGraph()
    G.add_nodes_from(pairs)

    if not flow_counts:
        return G

    max_count = max(flow_counts.values())

    for (src, tgt), count in flow_counts.items():
        avg_strength = np.mean(flow_strengths[(src, tgt)])
        G.add_edge(
            src, tgt,
            weight=count / max_count,  # 正規化
            count=count,
            avg_strength=round(avg_strength, 4),
        )

    return G


def calc_flow_metrics(G: nx.DiGraph) -> pd.DataFrame:
    """オーダーフローネットワークの指標を計算する.

    Args:
        G: 有向グラフ

    Returns:
        ノード別指標のDataFrame
    """
    if len(G.nodes()) == 0:
        return pd.DataFrame()

    in_degree = dict(G.in_degree(weight="weight"))
    out_degree = dict(G.out_degree(weight="weight"))

    # PageRank（情報発信源としての重要度）
    try:
        pagerank = nx.pagerank(G, weight="weight")
    except Exception:
        pagerank = {n: 1.0 / len(G.nodes()) for n in G.nodes()}

    rows = []
    for node in G.nodes():
        rows.append({
            "通貨ペア": node,
            "入次数（影響受け）": round(in_degree.get(node, 0), 3),
            "出次数（影響与え）": round(out_degree.get(node, 0), 3),
            "PageRank": round(pagerank.get(node, 0), 4),
            "影響力比": round(
                out_degree.get(node, 0) / max(in_degree.get(node, 0), 0.001),
                2,
            ),
        })

    return (
        pd.DataFrame(rows)
        .sort_values("PageRank", ascending=False)
        .reset_index(drop=True)
    )


def get_top_flows(G: nx.DiGraph, top_n: int = 10) -> pd.DataFrame:
    """最も強い伝播フローをリストする."""
    if len(G.edges()) == 0:
        return pd.DataFrame(columns=["from", "to", "count", "strength"])

    rows = []
    for u, v, data in G.edges(data=True):
        rows.append({
            "発信": u,
            "受信": v,
            "回数": data.get("count", 0),
            "平均反応(%)": data.get("avg_strength", 0),
            "重み": round(data.get("weight", 0), 3),
        })

    return (
        pd.DataFrame(rows)
        .sort_values("回数", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
