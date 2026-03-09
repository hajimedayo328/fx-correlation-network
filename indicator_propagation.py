"""経済指標別 波及パターン分析.

経済指標の発表タイミングと通貨ペアの価格変動を紐づけ、
指標ごとの波及パターンを分析する。

例: NFP発表 → USDJPYが最初に動く(0バー) → EURJPY(2バー遅れ) → GBPJPY(3バー遅れ)

グラフ理論的意味:
- 有向グラフ: 指標発表後の情報伝播構造
- ノード = 通貨ペア
- エッジ = 波及方向（先に動いたペア → 後に動いたペア）
- 重み = 波及の遅延（バー数）
"""

from __future__ import annotations

from datetime import datetime, timedelta

import networkx as nx
import numpy as np
import pandas as pd

from economic_calendar import get_indicator_short_name


def match_events_to_prices(
    calendar_df: pd.DataFrame,
    closes: pd.DataFrame,
    timeframe_minutes: int = 5,
    pre_bars: int = 6,
    post_bars: int = 24,
) -> list[dict]:
    """経済指標イベントと価格データをマッチングする.

    Args:
        calendar_df: カレンダーDataFrame（datetime列が必要）
        closes: 終値DataFrame（index=datetime, columns=通貨ペア）
        timeframe_minutes: 時間足（分）
        pre_bars: イベント前の取得バー数
        post_bars: イベント後の取得バー数

    Returns:
        マッチしたイベントのリスト
    """
    matched = []
    price_index = closes.index

    for _, event in calendar_df.iterrows():
        event_dt = event.get("datetime")
        if pd.isna(event_dt):
            continue

        # 価格データ中の最も近い時刻を検索
        if hasattr(event_dt, "timestamp"):
            event_ts = pd.Timestamp(event_dt)
        else:
            continue

        # 時間差が最小のインデックスを探す
        time_diffs = abs(price_index - event_ts)
        if len(time_diffs) == 0:
            continue

        closest_idx = time_diffs.argmin()
        min_diff = time_diffs[closest_idx]

        # 30分以上ずれていたらスキップ
        if min_diff > pd.Timedelta(minutes=30):
            continue

        # イベント前後の価格を取得
        start_idx = max(0, closest_idx - pre_bars)
        end_idx = min(len(price_index), closest_idx + post_bars + 1)

        pre_prices = closes.iloc[start_idx:closest_idx]
        post_prices = closes.iloc[closest_idx:end_idx]

        if len(post_prices) < 3:
            continue

        matched.append({
            "event": event["event"],
            "currency": event["currency"],
            "impact": event.get("impact", "High"),
            "datetime": event_dt,
            "actual": event.get("actual", ""),
            "forecast": event.get("forecast", ""),
            "previous": event.get("previous", ""),
            "price_idx": closest_idx,
            "pre_prices": pre_prices,
            "post_prices": post_prices,
        })

    return matched


def analyze_single_event_propagation(
    post_prices: pd.DataFrame,
    vol_threshold: float = 1.5,
) -> dict:
    """単一イベント後の波及パターンを分析する.

    Args:
        post_prices: イベント後の価格DataFrame
        vol_threshold: 「反応」と判定する閾値（平常時σの何倍か）

    Returns:
        各ペアの反応時刻・反応量・順序
    """
    if len(post_prices) < 2:
        return {}

    returns = post_prices.pct_change().dropna()
    if len(returns) == 0:
        return {}

    abs_returns = returns.abs()

    reactions = {}
    for pair in returns.columns:
        pair_returns = abs_returns[pair]

        # 最初の大きな反応バーを検出
        mean_ret = pair_returns.mean()
        std_ret = pair_returns.std()
        if std_ret == 0:
            continue

        threshold = mean_ret + vol_threshold * std_ret

        first_reaction_bar = None
        max_reaction = 0.0
        cumulative_move = 0.0

        for bar_idx, (t, ret) in enumerate(pair_returns.items()):
            if ret > threshold and first_reaction_bar is None:
                first_reaction_bar = bar_idx
            if ret > max_reaction:
                max_reaction = ret

        # 累積変動（方向あり）
        if len(returns) > 0:
            cumulative_move = returns[pair].sum()

        reactions[pair] = {
            "first_reaction_bar": first_reaction_bar if first_reaction_bar is not None else len(pair_returns),
            "max_reaction_pct": round(max_reaction * 100, 4),
            "cumulative_move_pct": round(cumulative_move * 100, 4),
            "direction": "up" if cumulative_move > 0 else "down",
        }

    return reactions


def analyze_indicator_propagation(
    matched_events: list[dict],
    indicator_name: str | None = None,
) -> pd.DataFrame:
    """指標別の波及パターンを集計する.

    Args:
        matched_events: match_events_to_prices()の結果
        indicator_name: 特定指標のみ分析（Noneなら全指標）

    Returns:
        指標×通貨ペアの波及統計
    """
    if indicator_name:
        events = [e for e in matched_events if e["event"] == indicator_name]
    else:
        events = matched_events

    if not events:
        return pd.DataFrame()

    # 指標ごとに集計
    indicator_stats = {}

    for event in events:
        name = event["event"]
        reactions = analyze_single_event_propagation(event["post_prices"])

        if not reactions:
            continue

        if name not in indicator_stats:
            indicator_stats[name] = {
                "count": 0,
                "pair_reactions": {},
            }

        indicator_stats[name]["count"] += 1

        for pair, react in reactions.items():
            if pair not in indicator_stats[name]["pair_reactions"]:
                indicator_stats[name]["pair_reactions"][pair] = {
                    "reaction_bars": [],
                    "max_reactions": [],
                    "cumulative_moves": [],
                }

            stats = indicator_stats[name]["pair_reactions"][pair]
            stats["reaction_bars"].append(react["first_reaction_bar"])
            stats["max_reactions"].append(react["max_reaction_pct"])
            stats["cumulative_moves"].append(react["cumulative_move_pct"])

    # DataFrameに変換
    rows = []
    for indicator, data in indicator_stats.items():
        for pair, stats in data["pair_reactions"].items():
            avg_reaction_bar = np.mean(stats["reaction_bars"])
            avg_max_reaction = np.mean(stats["max_reactions"])
            avg_cumulative = np.mean(stats["cumulative_moves"])

            rows.append({
                "指標": get_indicator_short_name(indicator),
                "指標(正式名)": indicator,
                "通貨ペア": pair,
                "発表回数": data["count"],
                "平均反応バー": round(avg_reaction_bar, 1),
                "平均最大反応(%)": round(avg_max_reaction, 4),
                "平均累積変動(%)": round(avg_cumulative, 4),
            })

    if not rows:
        return pd.DataFrame()

    return (
        pd.DataFrame(rows)
        .sort_values(["指標", "平均反応バー"])
        .reset_index(drop=True)
    )


def build_propagation_network(
    matched_events: list[dict],
    indicator_name: str,
    min_events: int = 2,
) -> nx.DiGraph:
    """特定指標の波及ネットワークを構築する.

    「先に動いたペア → 後に動いたペア」の有向グラフ。

    Args:
        matched_events: match_events_to_prices()の結果
        indicator_name: 対象指標名
        min_events: 最低イベント数

    Returns:
        有向グラフ
    """
    events = [e for e in matched_events if e["event"] == indicator_name]

    if len(events) < min_events:
        return nx.DiGraph()

    # 各イベントの反応順序を記録
    edge_lags: dict[tuple[str, str], list[int]] = {}
    edge_strengths: dict[tuple[str, str], list[float]] = {}

    for event in events:
        reactions = analyze_single_event_propagation(event["post_prices"])
        if not reactions:
            continue

        # 反応バーでソート → 先に動いたペア順
        sorted_pairs = sorted(
            reactions.items(),
            key=lambda x: x[1]["first_reaction_bar"],
        )

        for i, (src, src_react) in enumerate(sorted_pairs):
            for j, (tgt, tgt_react) in enumerate(sorted_pairs):
                if i >= j:
                    continue

                lag = tgt_react["first_reaction_bar"] - src_react["first_reaction_bar"]
                if lag <= 0:
                    continue

                key = (src, tgt)
                if key not in edge_lags:
                    edge_lags[key] = []
                    edge_strengths[key] = []
                edge_lags[key].append(lag)
                edge_strengths[key].append(tgt_react["max_reaction_pct"])

    # グラフ構築
    G = nx.DiGraph()

    all_pairs = set()
    for event in events:
        all_pairs.update(event["post_prices"].columns)
    G.add_nodes_from(all_pairs)

    for (src, tgt), lags in edge_lags.items():
        if len(lags) < max(1, min_events // 2):
            continue

        G.add_edge(
            src, tgt,
            avg_lag=round(np.mean(lags), 1),
            count=len(lags),
            avg_strength=round(np.mean(edge_strengths[(src, tgt)]), 4),
            weight=1.0 / max(np.mean(lags), 0.1),  # 遅延が小さいほど重い
        )

    return G


def get_propagation_order(
    matched_events: list[dict],
    indicator_name: str,
) -> pd.DataFrame:
    """特定指標の波及順序（平均）を取得する.

    Returns:
        通貨ペアの反応順位・平均反応時間・平均反応量
    """
    events = [e for e in matched_events if e["event"] == indicator_name]

    if not events:
        return pd.DataFrame()

    pair_stats: dict[str, list] = {}

    for event in events:
        reactions = analyze_single_event_propagation(event["post_prices"])
        for pair, react in reactions.items():
            if pair not in pair_stats:
                pair_stats[pair] = []
            pair_stats[pair].append(react)

    rows = []
    for pair, stats_list in pair_stats.items():
        avg_bar = np.mean([s["first_reaction_bar"] for s in stats_list])
        avg_max = np.mean([s["max_reaction_pct"] for s in stats_list])
        avg_cum = np.mean([s["cumulative_move_pct"] for s in stats_list])

        rows.append({
            "通貨ペア": pair,
            "平均反応順位(バー)": round(avg_bar, 1),
            "平均最大反応(%)": round(avg_max, 4),
            "平均累積変動(%)": round(avg_cum, 4),
            "サンプル数": len(stats_list),
        })

    return (
        pd.DataFrame(rows)
        .sort_values("平均反応順位(バー)")
        .reset_index(drop=True)
    )


def summarize_all_indicators(
    matched_events: list[dict],
) -> pd.DataFrame:
    """全指標のサマリーを作成する.

    Returns:
        指標ごとの影響度・最初に動くペア・平均反応
    """
    # 指標ごとにグループ化
    by_indicator: dict[str, list] = {}
    for event in matched_events:
        name = event["event"]
        if name not in by_indicator:
            by_indicator[name] = []
        by_indicator[name].append(event)

    rows = []
    for indicator, events in by_indicator.items():
        all_reactions = []
        first_movers = []

        for event in events:
            reactions = analyze_single_event_propagation(event["post_prices"])
            if not reactions:
                continue

            all_reactions.append(reactions)

            # 最初に動いたペア
            sorted_pairs = sorted(
                reactions.items(),
                key=lambda x: x[1]["first_reaction_bar"],
            )
            if sorted_pairs:
                first_movers.append(sorted_pairs[0][0])

        if not all_reactions:
            continue

        # 最頻の最初反応ペア
        if first_movers:
            from collections import Counter
            most_common_first = Counter(first_movers).most_common(1)[0]
        else:
            most_common_first = ("-", 0)

        # 全ペアの平均最大反応
        max_reactions = []
        for reactions in all_reactions:
            for react in reactions.values():
                max_reactions.append(react["max_reaction_pct"])

        rows.append({
            "指標": get_indicator_short_name(indicator),
            "指標(正式名)": indicator,
            "通貨": events[0]["currency"],
            "発表回数": len(events),
            "平均最大反応(%)": round(np.mean(max_reactions), 4) if max_reactions else 0,
            "最頻先行ペア": most_common_first[0],
            "先行回数": most_common_first[1],
            "影響ペア数": round(np.mean([len(r) for r in all_reactions]), 1),
        })

    if not rows:
        return pd.DataFrame()

    return (
        pd.DataFrame(rows)
        .sort_values("平均最大反応(%)", ascending=False)
        .reset_index(drop=True)
    )
