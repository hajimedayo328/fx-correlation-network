"""通貨相関ネットワーク リアルタイム可視化ダッシュボード.

Streamlit + Plotly で通貨間の相関構造をインタラクティブに表示する。
起動: streamlit run app.py
"""

from __future__ import annotations

import time

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from correlation import calc_correlation_matrix, filter_by_threshold
from data_fetcher import DEFAULT_SYMBOLS, TIMEFRAMES, fetch_multi_pair_closes
from dcc_garch import HAS_ARCH, calc_dcc_correlation, compare_correlation_methods
from graph_builder import (
    build_graph,
    build_mst,
    calc_algebraic_connectivity,
    calc_graph_metrics,
    calc_mst_metrics,
    calc_normalized_tree_length,
    calc_spectral_metrics,
    get_graph_summary,
    get_mst_summary,
)
from regime_classifier import (
    REGIME_COLORS,
    classify_rolling_regimes,
    get_regime_summary,
)
from rolling_correlation import (
    calc_rolling_all_metrics,
    calc_rolling_correlation,
    calc_rolling_graph_density,
    calc_rolling_ntl,
    detect_correlation_breakdowns,
)
from oikawa_backtest import (
    calc_stats,
    fetch_backtest_data,
    run_backtest,
    trades_to_dataframe,
)
from oikawa_strategy import OIKAWA_PAIRS, calc_currency_strength
from order_flow import (
    calc_flow_metrics,
    calc_volume_flow_network,
    get_top_flows,
)
from lead_lag import (
    build_lead_lag_network,
    calc_lead_lag_matrix,
    calc_lead_lag_metrics,
    detect_propagation_events,
)
from economic_calendar import (
    create_builtin_calendar,
    fetch_forex_factory_calendar,
    filter_calendar_by_currency,
    get_indicator_short_name,
)
from indicator_propagation import (
    analyze_indicator_propagation,
    build_propagation_network,
    get_propagation_order,
    match_events_to_prices,
    summarize_all_indicators,
)

# --- ページ設定 ---
st.set_page_config(
    page_title="通貨相関ネットワーク",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("通貨相関ネットワーク")
st.caption("グラフ理論 × FX — リアルタイム通貨間相関の可視化")

# --- タブ ---
tab_names = [
    "リアルタイム", "MST（最小全域木）", "構造分析",
    "過去検証", "DCC-GARCH", "及川式バックテスト",
    "オーダーフロー", "リードラグ", "指標波及分析",
]
(
    tab_realtime, tab_mst, tab_structure, tab_backtest,
    tab_dcc, tab_oikawa, tab_flow, tab_leadlag, tab_indicator,
) = st.tabs(tab_names)

# --- サイドバー ---
with st.sidebar:
    st.header("設定")

    timeframe = st.selectbox(
        "時間足",
        list(TIMEFRAMES.keys()),
        index=list(TIMEFRAMES.keys()).index("H1"),
    )

    count = st.slider("データ本数", min_value=10, max_value=500, value=100, step=10)

    threshold = st.slider(
        "相関閾値",
        min_value=0.0,
        max_value=0.9,
        value=0.3,
        step=0.05,
        help="この絶対値以上の相関のみエッジとして表示",
    )

    corr_method = st.selectbox(
        "相関計算手法",
        ["pearson", "spearman", "kendall"],
        index=0,
    )

    st.divider()
    st.subheader("過去検証設定")
    backtest_count = st.slider(
        "過去データ本数",
        min_value=100,
        max_value=5000,
        value=500,
        step=100,
        help="過去検証に使うデータ量（多いほど長期間）",
    )
    rolling_window = st.slider(
        "ローリングウィンドウ",
        min_value=10,
        max_value=200,
        value=50,
        step=10,
        help="相関を計算するウィンドウ幅（足数）",
    )

    st.divider()
    auto_refresh = st.checkbox("自動更新", value=False)
    refresh_interval = st.selectbox(
        "更新間隔",
        [30, 60, 120, 300],
        index=1,
        format_func=lambda x: f"{x}秒" if x < 60 else f"{x // 60}分",
    )

    if st.button("手動更新", type="primary", use_container_width=True):
        st.cache_data.clear()


# --- データ取得 ---
@st.cache_data(ttl=30)
def load_data(
    _symbols: tuple[str, ...],
    _timeframe: str,
    _count: int,
) -> pd.DataFrame | None:
    """通貨ペアデータを取得する（キャッシュ付き）."""
    try:
        return fetch_multi_pair_closes(
            symbols=list(_symbols),
            timeframe_key=_timeframe,
            count=_count,
        )
    except Exception as e:
        st.error(f"データ取得エラー: {e}")
        return None


# --- ネットワーク図（Plotly） ---
def create_network_figure(G: nx.Graph) -> go.Figure:
    """Plotlyでインタラクティブなネットワーク図を描画する."""
    pos = nx.spring_layout(G, seed=42, k=2.0)

    edge_traces = []
    for u, v, data in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        weight = data.get("weight", 0)
        abs_w = abs(weight)
        color = "rgba(220, 50, 50, 0.6)" if weight > 0 else "rgba(50, 50, 220, 0.6)"
        width = max(1, abs_w * 8)

        edge_traces.append(
            go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode="lines",
                line=dict(width=width, color=color),
                hoverinfo="text",
                text=f"{u} ↔ {v}: {weight:+.3f}",
                showlegend=False,
            )
        )

    node_x = [pos[n][0] for n in G.nodes()]
    node_y = [pos[n][1] for n in G.nodes()]
    node_degrees = [G.degree(n) for n in G.nodes()]
    node_text = [f"{n}<br>接続数: {G.degree(n)}" for n in G.nodes()]

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=list(G.nodes()),
        textposition="top center",
        textfont=dict(size=14, color="#333"),
        hovertext=node_text,
        hoverinfo="text",
        marker=dict(
            size=[max(25, d * 8 + 20) for d in node_degrees],
            color=node_degrees,
            colorscale="YlOrRd",
            showscale=True,
            colorbar=dict(title="接続数", thickness=15),
            line=dict(width=2, color="#333"),
        ),
        showlegend=False,
    )

    fig = go.Figure(data=[*edge_traces, node_trace])
    fig.update_layout(
        height=600,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor="white",
        annotations=[
            dict(
                text="赤線=正の相関 / 青線=負の相関 / 線の太さ=相関の強さ",
                x=0.5, y=-0.05, xref="paper", yref="paper",
                showarrow=False, font=dict(size=12, color="gray"),
            )
        ],
    )
    return fig


# --- MST図（Plotly） ---
def create_mst_figure(mst: nx.Graph) -> go.Figure:
    """PlotlyでMST（最小全域木）を描画する."""
    pos = nx.spring_layout(mst, seed=42, k=3.0)

    edge_traces = []
    for u, v, data in mst.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        corr = data.get("correlation", 0)
        dist = data.get("weight", 0)
        color = "rgba(197, 61, 67, 0.7)" if corr > 0 else "rgba(43, 60, 94, 0.7)"
        width = max(1.5, (2 - dist) * 4)

        edge_traces.append(
            go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode="lines",
                line=dict(width=width, color=color),
                hoverinfo="text",
                text=f"{u} ↔ {v}<br>相関: {corr:+.3f}<br>距離: {dist:.3f}",
                showlegend=False,
            )
        )

    degrees = dict(mst.degree())
    node_x = [pos[n][0] for n in mst.nodes()]
    node_y = [pos[n][1] for n in mst.nodes()]
    node_degrees = [degrees[n] for n in mst.nodes()]
    node_text = [
        f"{n}<br>MST次数: {degrees[n]}"
        for n in mst.nodes()
    ]

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=list(mst.nodes()),
        textposition="top center",
        textfont=dict(size=14, color="#333"),
        hovertext=node_text,
        hoverinfo="text",
        marker=dict(
            size=[max(30, d * 12 + 20) for d in node_degrees],
            color=node_degrees,
            colorscale="YlOrRd",
            showscale=True,
            colorbar=dict(title="MST次数", thickness=15),
            line=dict(width=2, color="#333"),
        ),
        showlegend=False,
    )

    fig = go.Figure(data=[*edge_traces, node_trace])
    fig.update_layout(
        height=600,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor="white",
        annotations=[
            dict(
                text="MST: 全通貨を最小コストで接続する木構造 / 赤=正の相関 / 青=負の相関",
                x=0.5, y=-0.05, xref="paper", yref="paper",
                showarrow=False, font=dict(size=12, color="gray"),
            )
        ],
    )
    return fig


# ========================================
# リアルタイムタブ
# ========================================
with tab_realtime:
    closes = load_data(
        _symbols=tuple(DEFAULT_SYMBOLS),
        _timeframe=timeframe,
        _count=count,
    )

    if closes is None:
        st.warning("MT5を起動してからページを更新してください。")
        st.stop()

    st.success(f"{len(closes.columns)}通貨ペア × {len(closes)}本のデータを取得")

    corr_matrix = calc_correlation_matrix(closes, method=corr_method)
    G = build_graph(corr_matrix, threshold=threshold)

    col1, col2 = st.columns([3, 2])

    with col1:
        st.subheader("ネットワーク図")
        fig = create_network_figure(G)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("相関行列ヒートマップ")
        heatmap = px.imshow(
            corr_matrix,
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
            aspect="equal",
            text_auto=".2f",
        )
        heatmap.update_layout(height=500, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(heatmap, use_container_width=True)

    st.subheader("グラフ理論指標")
    col_summary, col_metrics = st.columns([1, 3])

    with col_summary:
        summary = get_graph_summary(G)
        for key, val in summary.items():
            st.metric(label=key, value=val)

    with col_metrics:
        metrics = calc_graph_metrics(G)
        st.dataframe(metrics, use_container_width=True, hide_index=True)

    with st.expander("指標の解説（グラフ理論）"):
        st.markdown("""
        | 指標 | 意味 | FXでの解釈 |
        |------|------|------------|
        | **次数** | そのノードに接続しているエッジの数 | いくつの通貨ペアと強い相関を持っているか |
        | **次数中心性** | 次数 / (全ノード数-1)。0〜1で正規化 | ネットワーク全体の中でどれだけ中心的か |
        | **媒介中心性** | 最短経路上に位置する頻度 | 他の通貨ペア間の「橋渡し」になっている度合い |
        | **クラスタリング係数** | 隣接ノード同士がどれだけ繋がっているか | その通貨の周辺が「グループ」を形成しているか |
        | **グラフ密度** | 実際のエッジ数 / 可能なエッジ数 | 市場全体の相関がどれだけ強いか。1に近いほど全通貨が連動 |
        | **連結成分数** | 分離したグループの数 | 1なら全通貨が相関で繋がっている。2以上なら独立したグループがある |
        """)


# ========================================
# MSTタブ
# ========================================
with tab_mst:
    closes_mst = load_data(
        _symbols=tuple(DEFAULT_SYMBOLS),
        _timeframe=timeframe,
        _count=count,
    )

    if closes_mst is None:
        st.warning("MT5を起動してからページを更新してください。")
        st.stop()

    corr_mst = calc_correlation_matrix(closes_mst, method=corr_method)
    mst = build_mst(corr_mst)

    col_mst_graph, col_mst_info = st.columns([3, 2])

    with col_mst_graph:
        st.subheader("最小全域木（MST）")
        fig_mst = create_mst_figure(mst)
        st.plotly_chart(fig_mst, use_container_width=True)

    with col_mst_info:
        st.subheader("距離行列ヒートマップ")
        from graph_builder import _corr_to_distance
        dist_matrix = _corr_to_distance(corr_mst)
        heatmap_dist = px.imshow(
            dist_matrix,
            color_continuous_scale="Viridis_r",
            zmin=0, zmax=2,
            aspect="equal",
            text_auto=".2f",
        )
        heatmap_dist.update_layout(height=500, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(heatmap_dist, use_container_width=True)

    st.subheader("MST指標")
    col_mst_summary, col_mst_metrics = st.columns([1, 3])

    with col_mst_summary:
        mst_summary = get_mst_summary(mst)
        for key, val in mst_summary.items():
            st.metric(label=key, value=val)

    with col_mst_metrics:
        mst_metrics = calc_mst_metrics(mst)
        st.dataframe(mst_metrics, use_container_width=True, hide_index=True)

    with st.expander("MST指標の解説"):
        st.markdown("""
        | 指標 | 意味 | FXでの解釈 |
        |------|------|------------|
        | **MST次数** | MSTでの接続数 | 高い = その通貨がハブ（中心通貨）。市場の情報がこの通貨を経由して伝播 |
        | **MST媒介中心性** | MST上の最短経路を通る頻度 | 高い = 通貨グループ間の橋渡し。ここが動くと波及効果大 |
        | **葉ノード** | 末端ノード（次数1） | 市場構造の周辺に位置。独自の動きをしやすい |
        | **正規化木長(NTL)** | MSTエッジ距離の平均 | 低い = 市場全体が連動（危機時に収縮）。高い = バラバラ |
        | **ハブ通貨** | MST次数が最大の通貨 | 市場の中心。危機時にUSD系が中心化する傾向 |
        """)


# ========================================
# 構造分析タブ（Jaccard・Fiedler・レジーム分類）
# ========================================
with tab_structure:
    st.subheader("ネットワーク構造の時間変化分析")

    closes_struct = load_data(
        _symbols=tuple(DEFAULT_SYMBOLS),
        _timeframe=timeframe,
        _count=backtest_count,
    )

    if closes_struct is None:
        st.warning("MT5を起動してからページを更新してください。")
        st.stop()

    st.info(
        f"{len(closes_struct.columns)}通貨ペア × {len(closes_struct)}本 / "
        f"ウィンドウ: {rolling_window}本"
    )

    # 全指標を一括計算
    with st.spinner("全指標を計算中...（Jaccard・NTL・Fiedler・レジーム）"):
        all_metrics = calc_rolling_all_metrics(
            closes_struct, window=rolling_window, threshold=threshold
        )
        regimes = classify_rolling_regimes(all_metrics)

    # --- レジーム分類結果 ---
    st.subheader("市場レジーム自動分類")
    st.caption(
        "NTL・密度・Jaccard指数を組み合わせたルールベース分類 "
        "(Frontiers in Physics 2022)"
    )

    regime_summary = get_regime_summary(regimes)
    regime_cols = st.columns(len(regime_summary))
    for col, (regime, desc) in zip(regime_cols, regime_summary.items()):
        col.metric(label=regime, value=desc)

    # レジーム背景付きNTLチャート
    fig_regime = go.Figure()

    # レジームごとに色分けした背景
    for regime, color in REGIME_COLORS.items():
        mask = regimes == regime
        if mask.any():
            fig_regime.add_trace(
                go.Scatter(
                    x=all_metrics.index[mask],
                    y=all_metrics["NTL"][mask],
                    mode="markers",
                    marker=dict(color=color, size=4),
                    name=regime,
                )
            )

    fig_regime.add_trace(
        go.Scatter(
            x=all_metrics.index,
            y=all_metrics["NTL"],
            mode="lines",
            line=dict(color="rgba(0,0,0,0.3)", width=1),
            name="NTL",
            showlegend=False,
        )
    )
    fig_regime.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=20, b=20),
        yaxis=dict(title="NTL"),
        xaxis=dict(title="時刻"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_regime, use_container_width=True)

    # --- Jaccard指数の推移 ---
    st.subheader("Jaccard指数（MST安定性）")
    st.caption(
        "MDPI Entropy (2021): 1.0=構造が安定 / 急落=レジーム転換の先行シグナル"
    )

    fig_jaccard = go.Figure()
    fig_jaccard.add_trace(
        go.Scatter(
            x=all_metrics.index,
            y=all_metrics["Jaccard"],
            mode="lines",
            fill="tozeroy",
            line=dict(color="#E6B422", width=2),
            fillcolor="rgba(230, 180, 34, 0.2)",
            name="Jaccard",
        )
    )
    fig_jaccard.add_hline(
        y=0.5, line_dash="dash", line_color="red", opacity=0.7,
        annotation_text="転換警戒ライン",
    )
    fig_jaccard.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=20, b=20),
        yaxis=dict(title="Jaccard指数", range=[0, 1.05]),
        xaxis=dict(title="時刻"),
    )
    st.plotly_chart(fig_jaccard, use_container_width=True)

    # --- 代数的連結性の推移 ---
    st.subheader("代数的連結性（Fiedler値）")
    st.caption(
        "星野 (2025): ラプラシアン第2固有値。低下=ネットワークが分裂しやすい"
    )

    fig_alg = go.Figure()
    fig_alg.add_trace(
        go.Scatter(
            x=all_metrics.index,
            y=all_metrics["algebraic_connectivity"],
            mode="lines",
            fill="tozeroy",
            line=dict(color="#5B8930", width=2),
            fillcolor="rgba(91, 137, 48, 0.2)",
            name="代数的連結性",
        )
    )
    fig_alg.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=20, b=20),
        yaxis=dict(title="代数的連結性 (λ₂)"),
        xaxis=dict(title="時刻"),
    )
    st.plotly_chart(fig_alg, use_container_width=True)

    # --- ハブ通貨の変遷 ---
    st.subheader("ハブ通貨の変遷")
    st.caption(
        "Keskin et al. (2011): 危機時にUSD系通貨がハブに集中する傾向"
    )

    hub_counts = all_metrics["hub"].value_counts()
    fig_hub = px.bar(
        x=hub_counts.index,
        y=hub_counts.values,
        labels={"x": "通貨ペア", "y": "ハブ回数"},
        color=hub_counts.values,
        color_continuous_scale="YlOrRd",
    )
    fig_hub.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=20, b=20),
        showlegend=False,
    )
    st.plotly_chart(fig_hub, use_container_width=True)

    # Fiedlerベクトルによるクラスター分析（現在の相関行列）
    st.subheader("スペクトルクラスター分析（現在）")
    st.caption("Fiedlerベクトルの符号で通貨を2グループに自動分割")

    corr_struct = calc_correlation_matrix(closes_struct, method=corr_method)
    mst_struct = build_mst(corr_struct)
    spectral = calc_spectral_metrics(mst_struct)
    if not spectral.empty:
        col_a, col_b = st.columns(2)
        cluster_a = spectral[spectral["クラスター"] == "A"]
        cluster_b = spectral[spectral["クラスター"] == "B"]
        with col_a:
            st.markdown("**クラスターA**")
            st.dataframe(cluster_a, use_container_width=True, hide_index=True)
        with col_b:
            st.markdown("**クラスターB**")
            st.dataframe(cluster_b, use_container_width=True, hide_index=True)

    with st.expander("構造分析指標の解説"):
        st.markdown("""
        | 指標 | 論文 | 意味 | FXでの解釈 |
        |------|------|------|------------|
        | **Jaccard指数** | MDPI Entropy (2021) | 隣接MSTのエッジ類似度 | 1.0=安定、急落=構造変化。0.5以下は転換警戒 |
        | **代数的連結性** | 星野 (2025) | ラプラシアン第2固有値 | 低い=ネットワークが2分裂しやすい。急変は構造転換 |
        | **Fiedlerベクトル** | 星野 (2025) | 第2固有ベクトル | 符号でグラフを自然に2分割。通貨グループの境界を発見 |
        | **ハブ通貨** | Keskin (2011) | MST最大次数ノード | 危機時にUSD系に集中。平時は分散 |
        | **レジーム分類** | Frontiers (2022) | NTL+密度+Jaccardの複合判定 | 危機/トレンド/レンジ/転換期を自動分類 |
        """)


# ========================================
# 過去検証タブ
# ========================================
with tab_backtest:
    st.subheader("相関の時間変化（過去検証）")

    closes_bt = load_data(
        _symbols=tuple(DEFAULT_SYMBOLS),
        _timeframe=timeframe,
        _count=backtest_count,
    )

    if closes_bt is None:
        st.warning("MT5を起動してからページを更新してください。")
        st.stop()

    st.info(
        f"{len(closes_bt.columns)}通貨ペア × {len(closes_bt)}本 / "
        f"ウィンドウ: {rolling_window}本"
    )

    # --- 0. NTL（正規化木長）の推移 ---
    st.subheader("MST正規化木長（NTL）の推移")
    st.caption(
        "Onnela et al. (2003): NTL低下 = 市場が収縮（危機・強トレンド） / "
        "NTL上昇 = 通貨がバラバラに動く"
    )

    ntl_series = calc_rolling_ntl(closes_bt, window=rolling_window)

    fig_ntl = go.Figure()
    fig_ntl.add_trace(
        go.Scatter(
            x=ntl_series.index,
            y=ntl_series.values,
            mode="lines",
            fill="tozeroy",
            line=dict(color="#2B3C5E", width=2),
            fillcolor="rgba(43, 60, 94, 0.2)",
            name="NTL",
        )
    )
    fig_ntl.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=20, b=20),
        yaxis=dict(title="正規化木長"),
        xaxis=dict(title="時刻"),
    )
    st.plotly_chart(fig_ntl, use_container_width=True)

    # --- 1. グラフ密度の推移 ---
    st.subheader("グラフ密度の推移")
    st.caption("高い = 市場全体が連動（トレンド相場） / 低い = バラバラ（レンジ or 転換期）")

    density = calc_rolling_graph_density(
        closes_bt, window=rolling_window, threshold=threshold
    )

    fig_density = go.Figure()
    fig_density.add_trace(
        go.Scatter(
            x=density.index,
            y=density.values,
            mode="lines",
            fill="tozeroy",
            line=dict(color="#C53D43", width=2),
            fillcolor="rgba(197, 61, 67, 0.2)",
            name="グラフ密度",
        )
    )
    fig_density.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=20, b=20),
        yaxis=dict(title="密度", range=[0, 1]),
        xaxis=dict(title="時刻"),
    )
    st.plotly_chart(fig_density, use_container_width=True)

    # --- 2. ペア別ローリング相関 ---
    st.subheader("ペア別ローリング相関")

    rolling_corrs = calc_rolling_correlation(closes_bt, window=rolling_window)

    # ペア選択
    pair_options = list(rolling_corrs.keys())
    selected_pairs = st.multiselect(
        "表示するペア",
        pair_options,
        default=pair_options[:3] if len(pair_options) >= 3 else pair_options,
    )

    if selected_pairs:
        fig_rolling = go.Figure()
        colors = px.colors.qualitative.Set2
        for i, pair in enumerate(selected_pairs):
            series = rolling_corrs[pair].dropna()
            fig_rolling.add_trace(
                go.Scatter(
                    x=series.index,
                    y=series.values,
                    mode="lines",
                    name=pair,
                    line=dict(color=colors[i % len(colors)], width=2),
                )
            )

        fig_rolling.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig_rolling.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=20, b=20),
            yaxis=dict(title="相関係数", range=[-1, 1]),
            xaxis=dict(title="時刻"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig_rolling, use_container_width=True)

    # --- 3. 相関崩壊の検出 ---
    st.subheader("相関崩壊イベント")
    st.caption("短期間で相関が大きく変化したポイント = 市場構造の変化")

    drop_threshold = st.slider(
        "崩壊検出の閾値",
        min_value=0.1,
        max_value=0.8,
        value=0.3,
        step=0.05,
        help="この値以上の相関変化を「崩壊」として検出",
    )

    breakdowns = detect_correlation_breakdowns(
        rolling_corrs, drop_threshold=drop_threshold, window=rolling_window // 5
    )

    if breakdowns.empty:
        st.info("検出された相関崩壊イベントはありません。閾値を下げてみてください。")
    else:
        st.warning(f"{len(breakdowns)}件の相関崩壊イベントを検出")
        st.dataframe(
            breakdowns.rename(columns={
                "time": "時刻",
                "pair": "ペア",
                "change": "変化量",
                "before": "変化前",
                "after": "変化後",
            }),
            use_container_width=True,
            hide_index=True,
        )


# ========================================
# DCC-GARCHタブ
# ========================================
with tab_dcc:
    st.subheader("DCC-GARCH 動的相関")
    st.caption(
        "Musmeci et al. (2017): GARCH標準化残差の相関 = "
        "fat-tailに強い時変相関推定"
    )

    if not HAS_ARCH:
        st.error(
            "archパッケージが必要です: `pip install arch`"
        )
    else:
        closes_dcc = load_data(
            _symbols=tuple(DEFAULT_SYMBOLS),
            _timeframe=timeframe,
            _count=backtest_count,
        )

        if closes_dcc is None:
            st.warning("MT5を起動してからページを更新してください。")
            st.stop()

        symbols_dcc = closes_dcc.columns.tolist()

        col_p1, col_p2 = st.columns(2)
        with col_p1:
            pair1 = st.selectbox("通貨ペア1", symbols_dcc, index=0, key="dcc_p1")
        with col_p2:
            pair2 = st.selectbox(
                "通貨ペア2", symbols_dcc,
                index=min(1, len(symbols_dcc) - 1),
                key="dcc_p2",
            )

        if pair1 != pair2:
            with st.spinner("DCC-GARCH計算中..."):
                comparison = compare_correlation_methods(
                    closes_dcc, pair1, pair2, window=rolling_window
                )

            fig_dcc = go.Figure()
            fig_dcc.add_trace(
                go.Scatter(
                    x=comparison.index,
                    y=comparison["rolling"],
                    mode="lines",
                    name="ローリング相関",
                    line=dict(color="#2B3C5E", width=2),
                )
            )
            if "dcc" in comparison.columns:
                fig_dcc.add_trace(
                    go.Scatter(
                        x=comparison.index,
                        y=comparison["dcc"],
                        mode="lines",
                        name="DCC-GARCH相関",
                        line=dict(color="#E83929", width=2),
                    )
                )
            fig_dcc.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
            fig_dcc.update_layout(
                height=400,
                margin=dict(l=20, r=20, t=20, b=20),
                yaxis=dict(title="相関係数", range=[-1, 1]),
                xaxis=dict(title="時刻"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                title=f"{pair1} × {pair2}",
            )
            st.plotly_chart(fig_dcc, use_container_width=True)

            # 全ペアDCC相関
            st.subheader("全ペアDCC動的相関")
            with st.spinner("全ペアDCC計算中..."):
                dcc_corrs = calc_dcc_correlation(
                    closes_dcc, window=rolling_window
                )

            if dcc_corrs:
                dcc_pair_options = list(dcc_corrs.keys())
                selected_dcc = st.multiselect(
                    "表示するペア（DCC）",
                    dcc_pair_options,
                    default=dcc_pair_options[:3],
                    key="dcc_pairs",
                )

                if selected_dcc:
                    fig_dcc_all = go.Figure()
                    colors = px.colors.qualitative.Set2
                    for i, pair in enumerate(selected_dcc):
                        series = dcc_corrs[pair].dropna()
                        fig_dcc_all.add_trace(
                            go.Scatter(
                                x=series.index,
                                y=series.values,
                                mode="lines",
                                name=pair,
                                line=dict(
                                    color=colors[i % len(colors)], width=2
                                ),
                            )
                        )
                    fig_dcc_all.add_hline(
                        y=0, line_dash="dash", line_color="gray", opacity=0.5
                    )
                    fig_dcc_all.update_layout(
                        height=400,
                        margin=dict(l=20, r=20, t=20, b=20),
                        yaxis=dict(title="DCC相関係数", range=[-1, 1]),
                        xaxis=dict(title="時刻"),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02),
                    )
                    st.plotly_chart(fig_dcc_all, use_container_width=True)
            else:
                st.warning("DCC相関の計算に失敗しました。")
        else:
            st.warning("異なる通貨ペアを選択してください。")

    with st.expander("DCC-GARCHの解説"):
        st.markdown("""
        | 項目 | 説明 |
        |------|------|
        | **GARCH(1,1)** | 各通貨のボラティリティ変動をモデル化。fat-tail（裾の重い分布）に対応 |
        | **標準化残差** | GARCH推定後の残差を条件付き分散で割ったもの。ボラティリティ効果を除去 |
        | **DCC相関** | 標準化残差間のローリング相関。ボラティリティクラスタリングの影響を除いた「真の」相関変動 |
        | **vs ローリング相関** | ローリング相関はボラティリティの変動に引きずられやすい。DCC相関はより「構造的な」相関変化を捉える |
        | **用途** | 危機時の相関上昇がボラティリティ効果か構造変化かを区別できる |
        """)


# ========================================
# 及川式バックテストタブ
# ========================================
with tab_oikawa:
    st.subheader("及川式 通貨相関トレード バックテスト")
    st.caption(
        "通貨強弱 + デイリーピボットP0反発 + MA方向フィルター "
        "（及川圭哉氏の手法をベースに構築）"
    )

    with st.expander("戦略ロジック", expanded=False):
        st.markdown("""
        **エントリー条件（全て満たした場合のみ）:**
        1. **通貨強弱**: 10ペアのリターンから5通貨(EUR/GBP/USD/JPY/AUD)の強弱を算出
        2. **ペア選択**: 最強通貨 × 最弱通貨のペアを自動選択
        3. **ピボットP0接触**: 価格がデイリーピボット(前日H+L+C÷3)付近にある
        4. **MA方向フィルター**: 120MA(=1時間足10MA)の向きとトレード方向が一致
        5. **RR比**: リスクリワード1.0以上

        **決済:**
        - TP: 次のピボットレベル(R1 or S1)
        - SL: 反対側のピボットレベル + バッファ
        - タイムアウト: 最大保有バー数で強制決済
        """)

    # パラメータ設定
    st.subheader("パラメータ設定")
    col_p1, col_p2, col_p3 = st.columns(3)

    with col_p1:
        oikawa_count = st.slider(
            "データ本数（5分足）",
            min_value=1000, max_value=20000, value=5000, step=1000,
            key="oikawa_count",
            help="5000本 ≒ 約17日分",
        )
        oikawa_ma = st.slider(
            "MA期間", min_value=20, max_value=300, value=120, step=20,
            key="oikawa_ma",
            help="120 = 1時間足10MA相当（5分足上）",
        )

    with col_p2:
        oikawa_pivot_tol = st.slider(
            "ピボット接触許容(pips)",
            min_value=5.0, max_value=50.0, value=15.0, step=5.0,
            key="oikawa_pivot_tol",
        )
        oikawa_sl_buffer = st.slider(
            "SLバッファ(pips)",
            min_value=1.0, max_value=20.0, value=5.0, step=1.0,
            key="oikawa_sl_buf",
        )

    with col_p3:
        oikawa_strength_th = st.slider(
            "強弱閾値",
            min_value=0.0005, max_value=0.01, value=0.001, step=0.0005,
            key="oikawa_str_th",
            format="%.4f",
            help="通貨強弱スコア差の最低値",
        )
        oikawa_max_hold = st.slider(
            "最大保有バー数",
            min_value=12, max_value=120, value=60, step=12,
            key="oikawa_hold",
            help="60バー = 5時間",
        )

    # 実行ボタン
    if st.button("バックテスト実行", type="primary", key="run_oikawa"):
        with st.spinner("MT5からデータ取得中..."):
            try:
                m5_data, d1_data = fetch_backtest_data(count=oikawa_count)
                st.success(
                    f"取得完了: {len(m5_data)}ペア × M5 {oikawa_count}本"
                )
            except Exception as e:
                st.error(f"データ取得失敗: {e}")
                st.stop()

        with st.spinner("バックテスト実行中..."):
            trades = run_backtest(
                m5_data, d1_data,
                ma_period=oikawa_ma,
                pivot_tolerance_pips=oikawa_pivot_tol,
                sl_buffer_pips=oikawa_sl_buffer,
                strength_threshold=oikawa_strength_th,
                max_hold_bars=oikawa_max_hold,
            )
            stats = calc_stats(trades)

        # --- 結果表示 ---
        st.subheader("バックテスト結果")

        if stats["total"] == 0:
            st.warning("トレードが発生しませんでした。パラメータを調整してください。")
        else:
            # サマリー指標
            col_s1, col_s2, col_s3, col_s4, col_s5 = st.columns(5)
            col_s1.metric("トレード数", stats["total"])
            col_s2.metric("勝率", f"{stats['win_rate']}%")
            col_s3.metric("総Pips", f"{stats['total_pips']}")
            col_s4.metric("PF", f"{stats['profit_factor']}")
            col_s5.metric("最大DD", f"{stats['max_drawdown']} pips")

            col_s6, col_s7, col_s8, col_s9, col_s10 = st.columns(5)
            col_s6.metric("平均Pips", f"{stats['avg_pips']}")
            col_s7.metric("平均勝ち", f"{stats['avg_win']} pips")
            col_s8.metric("平均負け", f"{stats['avg_loss']} pips")
            col_s9.metric("最大連敗", stats["max_losing_streak"])
            col_s10.metric("平均保有", f"{stats['avg_hold_bars']}バー")

            # 決済タイプ別
            st.markdown(
                f"**決済内訳:** TP={stats['tp_count']} / "
                f"SL={stats['sl_count']} / "
                f"Timeout={stats['timeout_count']}"
            )

            # 損益カーブ
            st.subheader("損益カーブ")
            trade_df = trades_to_dataframe(trades)
            trade_df["cumulative_pips"] = trade_df["pips"].cumsum()

            fig_equity = go.Figure()
            fig_equity.add_trace(
                go.Scatter(
                    x=trade_df["entry_time"],
                    y=trade_df["cumulative_pips"],
                    mode="lines+markers",
                    line=dict(color="#2B3C5E", width=2),
                    marker=dict(
                        size=6,
                        color=[
                            "#5B8930" if p > 0 else "#E83929"
                            for p in trade_df["pips"]
                        ],
                    ),
                    name="累積Pips",
                )
            )
            fig_equity.add_hline(y=0, line_dash="dash", line_color="gray")
            fig_equity.update_layout(
                height=400,
                margin=dict(l=20, r=20, t=20, b=20),
                yaxis=dict(title="累積 Pips"),
                xaxis=dict(title="時刻"),
            )
            st.plotly_chart(fig_equity, use_container_width=True)

            # ペア別成績
            st.subheader("ペア別成績")
            pair_rows = []
            for pair, ps in stats["pair_stats"].items():
                wr = ps["wins"] / ps["trades"] * 100 if ps["trades"] > 0 else 0
                pair_rows.append({
                    "ペア": pair,
                    "トレード数": ps["trades"],
                    "勝率": f"{wr:.0f}%",
                    "総Pips": round(ps["pips"], 1),
                })
            st.dataframe(
                pd.DataFrame(pair_rows),
                use_container_width=True,
                hide_index=True,
            )

            # トレード一覧
            st.subheader("トレード一覧")
            display_df = trade_df[[
                "entry_time", "exit_time", "pair", "direction",
                "entry_price", "exit_price", "pips", "result",
                "strongest", "weakest", "hold_bars",
            ]].rename(columns={
                "entry_time": "エントリー",
                "exit_time": "決済",
                "pair": "ペア",
                "direction": "方向",
                "entry_price": "エントリー価格",
                "exit_price": "決済価格",
                "pips": "Pips",
                "result": "結果",
                "strongest": "最強通貨",
                "weakest": "最弱通貨",
                "hold_bars": "保有バー",
            })
            st.dataframe(display_df, use_container_width=True, hide_index=True)

            # 通貨強弱ヒートマップ（直近）
            st.subheader("通貨強弱の推移（参考）")
            st.caption("バックテスト期間中の各通貨の相対的な強さ")

            closes_dict_for_strength = {}
            for pair, m5 in m5_data.items():
                closes_dict_for_strength[pair] = m5[["close"]]
            strength_df = calc_currency_strength(
                closes_dict_for_strength, lookback=20
            )
            if not strength_df.empty:
                # 直近200本をサンプリング
                sample = strength_df.tail(200)
                fig_strength = go.Figure()
                colors_map = {
                    "EUR": "#2B3C5E", "GBP": "#C53D43",
                    "USD": "#5B8930", "JPY": "#E6B422",
                    "AUD": "#C47222",
                }
                for cur in sample.columns:
                    fig_strength.add_trace(
                        go.Scatter(
                            x=sample.index,
                            y=sample[cur],
                            mode="lines",
                            name=cur,
                            line=dict(
                                color=colors_map.get(cur, "gray"), width=2
                            ),
                        )
                    )
                fig_strength.add_hline(
                    y=0, line_dash="dash", line_color="gray", opacity=0.5
                )
                fig_strength.update_layout(
                    height=350,
                    margin=dict(l=20, r=20, t=20, b=20),
                    yaxis=dict(title="通貨強弱スコア"),
                    xaxis=dict(title="時刻"),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                )
                st.plotly_chart(fig_strength, use_container_width=True)


# ========================================
# オーダーフロータブ
# ========================================
with tab_flow:
    st.subheader("オーダーフローネットワーク")
    st.caption(
        "ティックボリュームのスパイクから売買圧の伝播構造を可視化。"
        "有向グラフ: A→B = Aでボリューム急増後にBが反応"
    )

    # データ取得（OHLCV必要）
    from data_fetcher import fetch_ohlcv_mt5, connect_mt5, disconnect_mt5

    flow_count = st.slider(
        "分析データ本数", min_value=200, max_value=5000,
        value=1000, step=200, key="flow_count",
    )
    flow_tf_key = st.selectbox(
        "時間足", ["M5", "M15", "H1"], index=0, key="flow_tf",
    )
    col_fp1, col_fp2 = st.columns(2)
    with col_fp1:
        flow_spike_th = st.slider(
            "スパイク閾値(σ)", 1.0, 4.0, 2.0, 0.5, key="flow_spike",
        )
    with col_fp2:
        flow_lag = st.slider(
            "伝播ラグ(バー)", 1, 10, 3, 1, key="flow_lag",
        )

    if st.button("オーダーフロー分析実行", type="primary", key="run_flow"):
        with st.spinner("OHLCV取得中..."):
            try:
                if not connect_mt5():
                    st.error("MT5接続失敗")
                    st.stop()
                ohlcv_dict = {}
                for pair in DEFAULT_SYMBOLS:
                    try:
                        df = fetch_ohlcv_mt5(
                            pair, TIMEFRAMES[flow_tf_key], flow_count
                        )
                        ohlcv_dict[pair] = df.set_index("time")
                    except Exception:
                        pass
                disconnect_mt5()
                st.success(f"{len(ohlcv_dict)}ペアのOHLCV取得完了")
            except Exception as e:
                st.error(f"データ取得失敗: {e}")
                st.stop()

        with st.spinner("オーダーフローネットワーク構築中..."):
            flow_G = calc_volume_flow_network(
                ohlcv_dict,
                spike_threshold=flow_spike_th,
                lag_bars=flow_lag,
            )

        if len(flow_G.edges()) == 0:
            st.warning("伝播フローが検出されませんでした。閾値を下げてみてください。")
        else:
            # 有向グラフ描画
            st.subheader("売買圧の伝播ネットワーク")
            pos = nx.spring_layout(flow_G, seed=42, k=2.5)

            edge_traces = []
            for u, v, data in flow_G.edges(data=True):
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                w = data.get("weight", 0.1)
                # 矢印をシミュレート（エッジの80%地点にマーカー）
                mx = x0 + 0.8 * (x1 - x0)
                my = y0 + 0.8 * (y1 - y0)
                edge_traces.append(
                    go.Scatter(
                        x=[x0, x1, None],
                        y=[y0, y1, None],
                        mode="lines",
                        line=dict(width=max(1, w * 6), color="rgba(197,61,67,0.5)"),
                        hoverinfo="text",
                        text=f"{u} → {v}<br>回数: {data.get('count',0)}<br>"
                             f"平均反応: {data.get('avg_strength',0)}%",
                        showlegend=False,
                    )
                )
                # 矢印の先端マーカー
                edge_traces.append(
                    go.Scatter(
                        x=[mx], y=[my],
                        mode="markers",
                        marker=dict(
                            symbol="triangle-right",
                            size=max(6, w * 12),
                            color="rgba(197,61,67,0.7)",
                            angle=np.degrees(np.arctan2(y1 - y0, x1 - x0)),
                        ),
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )

            # PageRankでノードサイズ
            try:
                pr = nx.pagerank(flow_G, weight="weight")
            except Exception:
                pr = {n: 0.1 for n in flow_G.nodes()}

            node_trace = go.Scatter(
                x=[pos[n][0] for n in flow_G.nodes()],
                y=[pos[n][1] for n in flow_G.nodes()],
                mode="markers+text",
                text=list(flow_G.nodes()),
                textposition="top center",
                textfont=dict(size=13),
                marker=dict(
                    size=[max(20, pr.get(n, 0.1) * 300) for n in flow_G.nodes()],
                    color=[pr.get(n, 0) for n in flow_G.nodes()],
                    colorscale="YlOrRd",
                    showscale=True,
                    colorbar=dict(title="PageRank"),
                    line=dict(width=2, color="#333"),
                ),
                hovertext=[
                    f"{n}<br>PageRank: {pr.get(n,0):.4f}"
                    for n in flow_G.nodes()
                ],
                hoverinfo="text",
                showlegend=False,
            )

            fig_flow = go.Figure(data=[*edge_traces, node_trace])
            fig_flow.update_layout(
                height=550,
                margin=dict(l=20, r=20, t=20, b=20),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                plot_bgcolor="white",
            )
            st.plotly_chart(fig_flow, use_container_width=True)

            # 指標テーブル
            col_fm, col_ft = st.columns([1, 1])
            with col_fm:
                st.subheader("フロー指標")
                flow_metrics = calc_flow_metrics(flow_G)
                st.dataframe(flow_metrics, use_container_width=True, hide_index=True)
            with col_ft:
                st.subheader("主要伝播フロー TOP10")
                top_flows = get_top_flows(flow_G)
                st.dataframe(top_flows, use_container_width=True, hide_index=True)

    with st.expander("オーダーフロー分析の解説"):
        st.markdown("""
        | 指標 | 意味 | FXでの解釈 |
        |------|------|------------|
        | **PageRank** | 情報発信源としての重要度 | 高い = 他の通貨に影響を波及させるハブ通貨 |
        | **出次数** | 他のペアに影響を与える頻度 | 高い = マーケットを先導するペア |
        | **入次数** | 他のペアから影響を受ける頻度 | 高い = 追随して動くペア |
        | **影響力比** | 出次数/入次数 | >1 = 先導型、<1 = 追随型 |
        | **伝播フロー** | A→Bの方向と頻度 | 大口がAで入った後にBが動くパターン |
        """)


# ========================================
# リードラグタブ
# ========================================
with tab_leadlag:
    st.subheader("リードラグ分析（経済指標波及ネットワーク）")
    st.caption(
        "クロス相関で通貨間の先行・遅行関係を検出。"
        "有向グラフ: A→B = Aの値動きがBに先行する"
    )

    closes_ll = load_data(
        _symbols=tuple(DEFAULT_SYMBOLS),
        _timeframe=timeframe,
        _count=backtest_count,
    )

    if closes_ll is None:
        st.warning("MT5を起動してからページを更新してください。")
        st.stop()

    col_ll1, col_ll2 = st.columns(2)
    with col_ll1:
        ll_max_lag = st.slider(
            "最大ラグ(バー)", 1, 30, 12, 1, key="ll_lag",
            help="クロス相関を検査するラグの範囲",
        )
    with col_ll2:
        ll_min_corr = st.slider(
            "最低相関", 0.05, 0.5, 0.15, 0.05, key="ll_corr",
            help="この相関以上の先行関係のみ表示",
        )

    # リードラグネットワーク構築
    with st.spinner("リードラグ分析中..."):
        ll_G = build_lead_lag_network(
            closes_ll, max_lag=ll_max_lag, min_correlation=ll_min_corr
        )
        lag_matrix, corr_ll_matrix = calc_lead_lag_matrix(
            closes_ll, max_lag=ll_max_lag
        )

    if len(ll_G.edges()) == 0:
        st.warning("先行関係が検出されませんでした。最低相関を下げてみてください。")
    else:
        # 有向グラフ描画
        st.subheader("先行・遅行ネットワーク")
        pos_ll = nx.spring_layout(ll_G, seed=42, k=2.5)

        edge_traces_ll = []
        for u, v, data in ll_G.edges(data=True):
            x0, y0 = pos_ll[u]
            x1, y1 = pos_ll[v]
            w = data.get("weight", 0.1)
            lag = data.get("lag", 0)
            corr = data.get("correlation", 0)
            color = "rgba(91,137,48,0.6)" if corr > 0 else "rgba(43,60,94,0.6)"

            edge_traces_ll.append(
                go.Scatter(
                    x=[x0, x1, None], y=[y0, y1, None],
                    mode="lines",
                    line=dict(width=max(1, w * 8), color=color),
                    hoverinfo="text",
                    text=f"{u} → {v}<br>ラグ: {lag}バー<br>相関: {corr}",
                    showlegend=False,
                )
            )
            mx = x0 + 0.8 * (x1 - x0)
            my = y0 + 0.8 * (y1 - y0)
            edge_traces_ll.append(
                go.Scatter(
                    x=[mx], y=[my],
                    mode="markers",
                    marker=dict(
                        symbol="triangle-right",
                        size=max(6, w * 12),
                        color=color,
                    ),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

        ll_metrics = calc_lead_lag_metrics(ll_G)
        lead_scores = {
            row["通貨ペア"]: row["先行スコア"]
            for _, row in ll_metrics.iterrows()
        }

        node_trace_ll = go.Scatter(
            x=[pos_ll[n][0] for n in ll_G.nodes()],
            y=[pos_ll[n][1] for n in ll_G.nodes()],
            mode="markers+text",
            text=list(ll_G.nodes()),
            textposition="top center",
            textfont=dict(size=13),
            marker=dict(
                size=30,
                color=[lead_scores.get(n, 0) for n in ll_G.nodes()],
                colorscale="RdYlGn",
                showscale=True,
                colorbar=dict(title="先行スコア"),
                line=dict(width=2, color="#333"),
            ),
            hovertext=[
                f"{n}<br>先行スコア: {lead_scores.get(n,0):.3f}"
                for n in ll_G.nodes()
            ],
            hoverinfo="text",
            showlegend=False,
        )

        fig_ll = go.Figure(data=[*edge_traces_ll, node_trace_ll])
        fig_ll.update_layout(
            height=550,
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor="white",
            annotations=[
                dict(
                    text="緑=正の相関で先行 / 青=負の相関で先行 / 矢印=先行方向",
                    x=0.5, y=-0.05, xref="paper", yref="paper",
                    showarrow=False, font=dict(size=12, color="gray"),
                )
            ],
        )
        st.plotly_chart(fig_ll, use_container_width=True)

    # リードラグ指標テーブル
    st.subheader("先行・遅行ランキング")
    st.caption("先行スコア = 出次数 - 入次数。正 = 先導型、負 = 追随型")
    if not ll_metrics.empty:
        st.dataframe(ll_metrics, use_container_width=True, hide_index=True)

    # ラグ行列ヒートマップ
    st.subheader("ラグ行列")
    st.caption("正の値 = 行の通貨が列の通貨に先行（バー数）")
    fig_lag = px.imshow(
        lag_matrix,
        color_continuous_scale="RdBu_r",
        zmin=-ll_max_lag, zmax=ll_max_lag,
        aspect="equal",
        text_auto=True,
    )
    fig_lag.update_layout(height=500, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig_lag, use_container_width=True)

    # 高ボラティリティイベント検出
    st.subheader("高ボラティリティイベント（経済指標等）")
    st.caption("複数ペアが同時にスパイクした時刻と伝播順序")
    events = detect_propagation_events(closes_ll)
    if events.empty:
        st.info("対象期間内に顕著なイベントは検出されませんでした。")
    else:
        st.dataframe(events, use_container_width=True, hide_index=True)

    with st.expander("リードラグ分析の解説"):
        st.markdown("""
        | 指標 | 意味 | FXでの解釈 |
        |------|------|------------|
        | **先行スコア** | 出次数-入次数 | 正=マーケットを先導、負=追随して動く |
        | **ラグ(バー)** | 先行する時間差 | 例: ラグ3（H1なら3時間先行） |
        | **クロス相関** | 時間差ありの相関 | 高い=先行関係が安定 |
        | **伝播イベント** | 複数ペア同時スパイク | 経済指標発表、地政学リスク等 |
        | **応用** | 先行ペアの動きを見てから遅行ペアをトレード | Granger因果性のグラフ版 |
        """)


# ============================================================
# Tab 9: 指標波及分析
# ============================================================
with tab_indicator:
    st.header("経済指標 波及パターン分析")
    st.caption(
        "経済指標の発表タイミングと通貨ペアの値動きを紐づけ、"
        "「どの指標が、どの通貨から、どの順番で波及するか」を分析する"
    )

    # --- 設定 ---
    col_ind1, col_ind2, col_ind3 = st.columns(3)
    with col_ind1:
        ind_data_source = st.radio(
            "カレンダーデータ",
            ["ビルトイン（デモ用）", "Forex Factory（スクレイピング）"],
            index=0,
            help="ビルトイン: 主要米国指標の典型的スケジュールを使用。"
                 "Forex Factory: 実際の発表日時を取得（時間がかかる場合あり）",
        )
    with col_ind2:
        ind_timeframe = st.selectbox(
            "価格データの時間足",
            ["M5", "M15", "H1"],
            index=0,
            key="ind_tf",
            help="M5: 5分足（高精度・データ少）, H1: 1時間足（低精度・データ多）",
        )
    with col_ind3:
        ind_count = st.slider(
            "取得本数", 500, 20000, 5000, step=500,
            key="ind_count",
            help="MT5から取得する価格データの本数",
        )

    ind_post_bars = st.slider(
        "発表後の分析バー数", 6, 48, 24, key="ind_post_bars",
        help="指標発表後、何バー分の値動きを分析するか",
    )

    if st.button("指標波及分析を実行", type="primary", key="btn_ind"):
        with st.spinner("カレンダーデータを取得中..."):
            if ind_data_source == "ビルトイン（デモ用）":
                calendar_df = create_builtin_calendar()
                calendar_df = filter_calendar_by_currency(calendar_df)
                st.info(
                    f"ビルトインカレンダー: {len(calendar_df)}件のイベント"
                    "（主要米国指標の典型的スケジュール）"
                )
            else:
                from datetime import datetime as dt
                try:
                    end = dt.now()
                    start = end - pd.Timedelta(days=90)
                    calendar_df = fetch_forex_factory_calendar(start, end)
                    calendar_df = filter_calendar_by_currency(calendar_df)
                    st.success(
                        f"Forex Factoryから {len(calendar_df)}件のイベントを取得"
                    )
                except Exception as e:
                    st.error(f"Forex Factory取得エラー: {e}")
                    st.info("ビルトインカレンダーにフォールバックします")
                    calendar_df = create_builtin_calendar()
                    calendar_df = filter_calendar_by_currency(calendar_df)

        # カレンダー表示
        with st.expander("カレンダーデータ（取得結果）", expanded=False):
            st.dataframe(calendar_df, use_container_width=True, hide_index=True)

        # 価格データ取得
        with st.spinner("MT5から価格データを取得中..."):
            try:
                closes_ind = fetch_multi_pair_closes(
                    symbols=DEFAULT_SYMBOLS,
                    timeframe_key=ind_timeframe,
                    count=ind_count,
                )
                st.success(
                    f"価格データ: {len(closes_ind)}本 × {len(closes_ind.columns)}ペア"
                )
            except Exception as e:
                st.error(f"価格データ取得エラー: {e}")
                st.stop()

        # イベントマッチング
        with st.spinner("イベントと価格データをマッチング中..."):
            tf_minutes = {"M5": 5, "M15": 15, "H1": 60}[ind_timeframe]
            matched = match_events_to_prices(
                calendar_df, closes_ind,
                timeframe_minutes=tf_minutes,
                post_bars=ind_post_bars,
            )
            st.info(f"マッチしたイベント: {len(matched)}件")

        if not matched:
            st.warning(
                "カレンダーのイベント日時と価格データの期間が重なっていません。\n"
                "取得本数を増やすか、Forex Factoryから実際の日時を取得してください。"
            )
        else:
            # --- 全指標サマリー ---
            st.subheader("指標別 影響度サマリー")
            summary = summarize_all_indicators(matched)
            if not summary.empty:
                st.dataframe(summary, use_container_width=True, hide_index=True)

                # 影響度の棒グラフ
                fig_impact = px.bar(
                    summary.head(15),
                    x="指標",
                    y="平均最大反応(%)",
                    color="通貨",
                    color_discrete_map={
                        "USD": "#2B3C5E",
                        "EUR": "#C53D43",
                        "GBP": "#5B8930",
                        "JPY": "#E6B422",
                        "AUD": "#C47222",
                    },
                    title="指標別 平均最大反応",
                )
                fig_impact.update_layout(height=400)
                st.plotly_chart(fig_impact, use_container_width=True)

            # --- 指標別波及分析 ---
            st.subheader("指標別 波及パターン詳細")

            # マッチしたイベントの指標名リスト
            indicator_names = sorted(set(e["event"] for e in matched))
            selected_indicator = st.selectbox(
                "分析する指標を選択",
                indicator_names,
                format_func=get_indicator_short_name,
                key="sel_indicator",
            )

            if selected_indicator:
                short_name = get_indicator_short_name(selected_indicator)
                st.markdown(f"### {short_name}（{selected_indicator}）")

                # 波及順序
                prop_order = get_propagation_order(matched, selected_indicator)
                if not prop_order.empty:
                    col_order1, col_order2 = st.columns([1, 1])

                    with col_order1:
                        st.markdown("**波及順序（平均）**")
                        st.dataframe(
                            prop_order, use_container_width=True, hide_index=True,
                        )

                        # 波及順序テキスト
                        order_str = " → ".join(
                            f"{row['通貨ペア']}({row['平均反応順位(バー)']})"
                            for _, row in prop_order.head(5).iterrows()
                        )
                        st.markdown(f"**波及チェーン:** {order_str}")

                    with col_order2:
                        # 反応バーの棒グラフ
                        fig_order = px.bar(
                            prop_order,
                            x="通貨ペア",
                            y="平均反応順位(バー)",
                            color="平均最大反応(%)",
                            color_continuous_scale="YlOrRd",
                            title=f"{short_name} 発表後の反応順序",
                        )
                        fig_order.update_layout(height=400)
                        st.plotly_chart(fig_order, use_container_width=True)

                # 波及ネットワーク
                prop_G = build_propagation_network(
                    matched, selected_indicator, min_events=1,
                )

                if len(prop_G.edges()) > 0:
                    st.markdown("**波及ネットワーク（有向グラフ）**")
                    st.caption(
                        "矢印: 先に動いたペア → 後に動いたペア / "
                        "エッジの太さ: 伝播頻度"
                    )

                    pos_prop = nx.spring_layout(prop_G, seed=42, k=2)

                    edge_traces_prop = []
                    for u, v, d in prop_G.edges(data=True):
                        x0, y0 = pos_prop[u]
                        x1, y1 = pos_prop[v]
                        count = d.get("count", 1)
                        avg_lag = d.get("avg_lag", 0)

                        edge_traces_prop.append(
                            go.Scatter(
                                x=[x0, x1, None],
                                y=[y0, y1, None],
                                mode="lines",
                                line=dict(
                                    width=max(1, count * 2),
                                    color="#C53D43",
                                ),
                                hoverinfo="text",
                                text=(
                                    f"{u} → {v}<br>"
                                    f"平均ラグ: {avg_lag}バー<br>"
                                    f"回数: {count}"
                                ),
                                showlegend=False,
                            )
                        )
                        # 矢印
                        mx = x0 + 0.75 * (x1 - x0)
                        my = y0 + 0.75 * (y1 - y0)
                        edge_traces_prop.append(
                            go.Scatter(
                                x=[mx], y=[my],
                                mode="markers",
                                marker=dict(
                                    symbol="triangle-right",
                                    size=max(6, count * 3),
                                    color="#C53D43",
                                ),
                                hoverinfo="skip",
                                showlegend=False,
                            )
                        )

                    # ノード
                    node_sizes = []
                    for n in prop_G.nodes():
                        out_d = prop_G.out_degree(n)
                        node_sizes.append(20 + out_d * 5)

                    node_trace_prop = go.Scatter(
                        x=[pos_prop[n][0] for n in prop_G.nodes()],
                        y=[pos_prop[n][1] for n in prop_G.nodes()],
                        mode="markers+text",
                        text=list(prop_G.nodes()),
                        textposition="top center",
                        textfont=dict(size=13),
                        marker=dict(
                            size=node_sizes,
                            color=[
                                prop_G.out_degree(n, weight="count")
                                for n in prop_G.nodes()
                            ],
                            colorscale="YlOrRd",
                            showscale=True,
                            colorbar=dict(title="発信回数"),
                            line=dict(width=2, color="#333"),
                        ),
                        hoverinfo="text",
                        showlegend=False,
                    )

                    fig_prop = go.Figure(
                        data=[*edge_traces_prop, node_trace_prop]
                    )
                    fig_prop.update_layout(
                        height=500,
                        margin=dict(l=20, r=20, t=20, b=20),
                        xaxis=dict(
                            showgrid=False, zeroline=False,
                            showticklabels=False,
                        ),
                        yaxis=dict(
                            showgrid=False, zeroline=False,
                            showticklabels=False,
                        ),
                        plot_bgcolor="white",
                    )
                    st.plotly_chart(fig_prop, use_container_width=True)
                else:
                    st.info("波及ネットワークのエッジが不足しています。")

            # --- 詳細テーブル ---
            with st.expander("全指標×全ペア 波及統計"):
                detail = analyze_indicator_propagation(matched)
                if not detail.empty:
                    st.dataframe(
                        detail, use_container_width=True, hide_index=True,
                    )

    with st.expander("指標波及分析の解説"):
        st.markdown("""
        | 概念 | 説明 |
        |------|------|
        | **波及パターン** | 指標発表後、どの通貨ペアがどの順番で動くか |
        | **反応バー** | 発表から最初の大きな値動きまでのバー数 |
        | **最大反応(%)** | 発表後の最大瞬間変動率 |
        | **累積変動(%)** | 発表後の合計変動（方向あり） |
        | **波及ネットワーク** | 先行ペア→遅行ペアの有向グラフ |
        | **応用** | NFP発表→USDJPYが最初に動く→数バー後にEURJPYが追随→そこを狙う |

        **データソース:**
        - ビルトイン: 主要米国指標（NFP, CPI, FOMC等）の典型的スケジュール
        - Forex Factory: 実際の発表日時・結果をスクレイピング

        **注意:** ビルトインカレンダーは典型的な日付の近似値です。
        正確な分析にはForex Factoryからの実データ取得を推奨します。
        """)


# --- 自動更新 ---
if auto_refresh:
    time.sleep(refresh_interval)
    st.cache_data.clear()
    st.rerun()
