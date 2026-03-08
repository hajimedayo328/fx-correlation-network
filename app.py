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
from graph_builder import build_graph, calc_graph_metrics, get_graph_summary
from rolling_correlation import (
    calc_rolling_correlation,
    calc_rolling_graph_density,
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
tab_realtime, tab_backtest = st.tabs(["リアルタイム", "過去検証"])

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


# --- 自動更新 ---
if auto_refresh:
    time.sleep(refresh_interval)
    st.cache_data.clear()
    st.rerun()
