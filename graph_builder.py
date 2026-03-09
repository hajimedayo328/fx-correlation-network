"""networkxによるグラフ構築とグラフ理論指標の計算.

相関行列から通貨ネットワーク・MST（最小全域木）を構築し、
中心性やクラスタリング係数を算出する。

参考論文:
- Mantegna (1999): 距離変換 d(i,j) = sqrt(2 * (1 - corr(i,j)))
- Onnela et al. (2003): 正規化木長(NTL)による危機検出
- MDPI Entropy (2021): Jaccard指数によるMST安定性評価
- 星野 (2025): Fiedlerベクトル・代数的連結性による構造変化検知
- Keskin et al. (2011): ハブ通貨の中心化現象
"""

from __future__ import annotations

import numpy as np
import networkx as nx
import pandas as pd
from scipy import linalg

from correlation import get_edge_list


def build_graph(
    corr_matrix: pd.DataFrame,
    threshold: float = 0.3,
) -> nx.Graph:
    """相関行列からnetworkxグラフを構築する.

    Args:
        corr_matrix: 通貨ペア間の相関行列
        threshold: エッジ生成の相関閾値

    Returns:
        ノード=通貨ペア、エッジ=相関のグラフ
    """
    G = nx.Graph()

    # 全通貨ペアをノードとして追加（相関が弱くても表示される）
    for symbol in corr_matrix.columns:
        G.add_node(symbol)

    # 閾値以上の相関をエッジとして追加
    edges = get_edge_list(corr_matrix, threshold)
    for edge in edges:
        G.add_edge(
            edge["source"],
            edge["target"],
            weight=edge["weight"],
            abs_weight=abs(edge["weight"]),
        )

    return G


def calc_graph_metrics(G: nx.Graph) -> pd.DataFrame:
    """グラフ理論の基本指標を計算する.

    Args:
        G: 通貨ネットワークグラフ

    Returns:
        各ノードの指標を含むDataFrame
    """
    # 次数中心性: どの通貨が最も多くの相関を持つか
    degree_centrality = nx.degree_centrality(G)

    # 媒介中心性: どの通貨がネットワークの橋渡し役か
    betweenness = nx.betweenness_centrality(G)

    # クラスタリング係数: 近傍ノード間の結合密度
    clustering = nx.clustering(G)

    # 次数（接続エッジ数）
    degree = dict(G.degree())

    metrics = pd.DataFrame({
        "通貨ペア": list(G.nodes()),
        "次数": [degree.get(n, 0) for n in G.nodes()],
        "次数中心性": [round(degree_centrality.get(n, 0), 4) for n in G.nodes()],
        "媒介中心性": [round(betweenness.get(n, 0), 4) for n in G.nodes()],
        "クラスタリング係数": [round(clustering.get(n, 0), 4) for n in G.nodes()],
    })

    return metrics.sort_values("次数中心性", ascending=False).reset_index(drop=True)


def get_graph_summary(G: nx.Graph) -> dict:
    """グラフ全体のサマリー情報を取得する.

    Returns:
        ノード数、エッジ数、密度、連結成分数
    """
    return {
        "ノード数": G.number_of_nodes(),
        "エッジ数": G.number_of_edges(),
        "グラフ密度": round(nx.density(G), 4),
        "連結成分数": nx.number_connected_components(G),
    }


# =============================================
# MST（最小全域木）
# =============================================


def _corr_to_distance(corr_matrix: pd.DataFrame) -> pd.DataFrame:
    """相関行列をMantegna距離行列に変換する.

    d(i,j) = sqrt(2 * (1 - corr(i,j)))
    相関1.0 → 距離0、相関0 → 距離√2、相関-1 → 距離2

    Args:
        corr_matrix: 相関行列

    Returns:
        距離行列（DataFrame）
    """
    return np.sqrt(2 * (1 - corr_matrix))


def build_mst(corr_matrix: pd.DataFrame) -> nx.Graph:
    """相関行列からMST（最小全域木）を構築する.

    Mantegna (1999) の手法に基づき、相関係数をユークリッド距離に変換し、
    最小全域木を構築する。閾値は不要（全ノードが必ず接続される）。

    Args:
        corr_matrix: 通貨ペア間の相関行列

    Returns:
        MST（nx.Graph）。エッジ属性にdistance, correlationを持つ
    """
    dist_matrix = _corr_to_distance(corr_matrix)
    symbols = corr_matrix.columns.tolist()

    # 完全グラフを距離で構築
    G_full = nx.Graph()
    for i, src in enumerate(symbols):
        for j, tgt in enumerate(symbols):
            if i >= j:
                continue
            d = dist_matrix.iloc[i, j]
            c = corr_matrix.iloc[i, j]
            G_full.add_edge(src, tgt, weight=d, correlation=c)

    # 最小全域木を抽出
    mst = nx.minimum_spanning_tree(G_full, weight="weight")
    return mst


def calc_normalized_tree_length(mst: nx.Graph) -> float:
    """正規化木長（NTL）を計算する.

    NTL = MST全エッジの距離合計 / (ノード数 - 1)
    低い → 市場全体が連動（危機時に収縮する傾向）
    高い → 通貨がバラバラに動いている

    Args:
        mst: MST グラフ

    Returns:
        正規化木長
    """
    n = mst.number_of_nodes()
    if n <= 1:
        return 0.0
    total = sum(d["weight"] for _, _, d in mst.edges(data=True))
    return round(total / (n - 1), 4)


def calc_mst_metrics(mst: nx.Graph) -> pd.DataFrame:
    """MST固有の指標を計算する.

    Args:
        mst: MST グラフ

    Returns:
        各ノードの次数・媒介中心性・葉ノードかどうか
    """
    degree = dict(mst.degree())
    betweenness = nx.betweenness_centrality(mst)

    metrics = pd.DataFrame({
        "通貨ペア": list(mst.nodes()),
        "MST次数": [degree.get(n, 0) for n in mst.nodes()],
        "MST媒介中心性": [round(betweenness.get(n, 0), 4) for n in mst.nodes()],
        "葉ノード": ["○" if degree.get(n, 0) == 1 else "" for n in mst.nodes()],
    })

    return metrics.sort_values("MST次数", ascending=False).reset_index(drop=True)


def get_mst_summary(mst: nx.Graph) -> dict:
    """MSTのサマリー情報を取得する.

    Returns:
        ノード数、エッジ数、正規化木長、ハブノード
    """
    degree = dict(mst.degree())
    hub = max(degree, key=degree.get) if degree else "N/A"
    hub_degree = degree.get(hub, 0)

    return {
        "ノード数": mst.number_of_nodes(),
        "エッジ数": mst.number_of_edges(),
        "正規化木長(NTL)": calc_normalized_tree_length(mst),
        "ハブ通貨": f"{hub} (次数{hub_degree})",
    }


# =============================================
# Jaccard指数（MST安定性）
# =============================================


def calc_mst_jaccard(mst1: nx.Graph, mst2: nx.Graph) -> float:
    """2つのMST間のJaccard類似度を計算する.

    J(t, t+1) = |E(t) ∩ E(t+1)| / |E(t) ∪ E(t+1)|
    1.0 = 完全に同じ構造、0.0 = 完全に異なる構造
    急落 = レジーム転換の先行シグナル (MDPI Entropy 2021)

    Args:
        mst1: 時刻tのMST
        mst2: 時刻t+1のMST

    Returns:
        Jaccard類似度 (0〜1)
    """
    edges1 = {frozenset(e) for e in mst1.edges()}
    edges2 = {frozenset(e) for e in mst2.edges()}

    intersection = len(edges1 & edges2)
    union = len(edges1 | edges2)

    if union == 0:
        return 1.0
    return round(intersection / union, 4)


# =============================================
# Fiedlerベクトル・代数的連結性（スペクトル分析）
# =============================================


def calc_algebraic_connectivity(G: nx.Graph) -> float:
    """代数的連結性（Fiedler値）を計算する.

    ラプラシアン行列の第2最小固有値。
    低い → グラフが分裂しやすい（弱い結合）
    高い → グラフが堅固に結合している

    星野 (2025) の手法: この値の急変が構造変化の指標

    Args:
        G: ネットワークグラフ

    Returns:
        代数的連結性（λ2）
    """
    if G.number_of_nodes() < 2:
        return 0.0
    if not nx.is_connected(G):
        # 非連結の場合、最大連結成分で計算
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
        if G.number_of_nodes() < 2:
            return 0.0

    return round(nx.algebraic_connectivity(G, method="tracemin_lu"), 6)


def calc_fiedler_vector(G: nx.Graph) -> dict[str, float]:
    """Fiedlerベクトル（第2最小固有ベクトル）を計算する.

    各ノードのFiedler値の符号でグラフを2分割できる。
    正のグループと負のグループ = 自然なクラスター構造

    Args:
        G: ネットワークグラフ

    Returns:
        {ノード名: Fiedler値} の辞書
    """
    if G.number_of_nodes() < 2:
        return {}
    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
        if G.number_of_nodes() < 2:
            return {}

    fiedler = nx.fiedler_vector(G, method="tracemin_lu")
    nodes = list(G.nodes())
    return {nodes[i]: round(float(fiedler[i]), 4) for i in range(len(nodes))}


def calc_spectral_metrics(G: nx.Graph) -> pd.DataFrame:
    """スペクトル指標（Fiedlerベクトル + クラスター分類）を計算する.

    Args:
        G: ネットワークグラフ

    Returns:
        各ノードのFiedler値とクラスター分類
    """
    fiedler = calc_fiedler_vector(G)
    if not fiedler:
        return pd.DataFrame(columns=["通貨ペア", "Fiedler値", "クラスター"])

    return pd.DataFrame({
        "通貨ペア": list(fiedler.keys()),
        "Fiedler値": list(fiedler.values()),
        "クラスター": ["A" if v >= 0 else "B" for v in fiedler.values()],
    }).sort_values("Fiedler値", ascending=False).reset_index(drop=True)


def get_mst_hub(mst: nx.Graph) -> str:
    """MSTのハブ通貨（最大次数ノード）を返す."""
    degree = dict(mst.degree())
    if not degree:
        return "N/A"
    return max(degree, key=degree.get)
