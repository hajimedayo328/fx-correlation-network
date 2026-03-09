"""networkxによるグラフ構築とグラフ理論指標の計算.

相関行列から通貨ネットワーク・MST（最小全域木）を構築し、
中心性やクラスタリング係数を算出する。

MST参考: Mantegna (1999) "Hierarchical Structure in Financial Markets"
距離変換: d(i,j) = sqrt(2 * (1 - corr(i,j)))
"""

from __future__ import annotations

import numpy as np
import networkx as nx
import pandas as pd

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
