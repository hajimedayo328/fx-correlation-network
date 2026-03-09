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

# --- ページ設定 ---
st.set_page_config(
    page_title="通貨相関ネットワーク",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("通貨相関ネットワーク")
st.caption("グラフ理論 × FX — リアルタイム通貨間相関の可視化")

# --- タブ ---
tab_names = ["リアルタイム", "MST（最小全域木）", "構造分析", "過去検証", "DCC-GARCH"]
tab_realtime, tab_mst, tab_structure, tab_backtest, tab_dcc = st.tabs(tab_names)

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


# --- 自動更新 ---
if auto_refresh:
    time.sleep(refresh_interval)
    st.cache_data.clear()
    st.rerun()
