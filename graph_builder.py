"""networkxによるグラフ構築とグラフ理論指標の計算.

相関行列から通貨ネットワークを構築し、中心性やクラスタリング係数を算出する。
"""

from __future__ import annotations

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
