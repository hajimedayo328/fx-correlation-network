"""及川式通貨相関トレード バックテストエンジン.

10通貨ペアの5分足データを使用し、
通貨強弱 + ピボットP0反発 + MA方向フィルターの戦略をバックテストする。

手順:
1. MT5から10ペアの5分足 + 日足データを取得
2. 日足からデイリーピボットを計算
3. 各5分足バーで通貨強弱を判定
4. 最強×最弱ペアがピボットP0付近 + MA方向一致ならエントリー
5. SL/TPヒットまたは時間切れで決済
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from data_fetcher import (
    TIMEFRAMES,
    connect_mt5,
    disconnect_mt5,
    fetch_ohlcv_mt5,
)
from oikawa_strategy import (
    OIKAWA_PAIRS,
    PAIR_CURRENCIES,
    OikawaSignal,
    calc_currency_strength,
    calc_daily_pivots,
    calc_sl_tp,
    check_ma_direction,
    find_strongest_weakest,
    get_direction_for_pair,
    get_pair_for_currencies,
    is_near_pivot,
    _get_pip_value,
)


# data_fetcherからimportを修正
import sys
from pathlib import Path

_BACKTEST_DIR = Path(__file__).resolve().parent.parent / "backtest"
if str(_BACKTEST_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKTEST_DIR))

from data.fetcher import connect_mt5, disconnect_mt5, fetch_ohlcv_mt5


@dataclass(frozen=True)
class TradeResult:
    """個別トレード結果."""

    signal: OikawaSignal
    exit_time: pd.Timestamp
    exit_price: float
    pips: float
    result: str  # "TP", "SL", "timeout"
    hold_bars: int


def fetch_backtest_data(
    count: int = 5000,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """バックテスト用データを取得する.

    Args:
        count: 5分足の本数

    Returns:
        (m5_data, d1_data) の辞書ペア
        m5_data: {ペア名: DataFrame(time,open,high,low,close)}
        d1_data: {ペア名: DataFrame(time,open,high,low,close)}
    """
    if not connect_mt5():
        raise RuntimeError("MT5接続失敗")

    try:
        m5_data = {}
        d1_data = {}

        for pair in OIKAWA_PAIRS:
            try:
                m5 = fetch_ohlcv_mt5(pair, TIMEFRAMES["M5"], count)
                m5_data[pair] = m5.set_index("time")

                # 日足は十分な過去データを取得
                d1 = fetch_ohlcv_mt5(pair, TIMEFRAMES["D1"], 100)
                d1_data[pair] = d1.set_index("time")
            except Exception as e:
                print(f"  {pair} 取得失敗: {e}")

        return m5_data, d1_data
    finally:
        disconnect_mt5()


def _get_daily_pivot_for_time(
    pivot_df: pd.DataFrame,
    current_time: pd.Timestamp,
) -> pd.Series | None:
    """現在時刻に対応する前日のピボットを返す."""
    current_date = current_time.normalize()

    # 前日以前のピボットを探す
    past_pivots = pivot_df[pivot_df.index < current_date]
    if len(past_pivots) == 0:
        return None
    return past_pivots.iloc[-1]


def run_backtest(
    m5_data: dict[str, pd.DataFrame],
    d1_data: dict[str, pd.DataFrame],
    ma_period: int = 120,
    pivot_tolerance_pips: float = 15.0,
    sl_buffer_pips: float = 5.0,
    strength_threshold: float = 0.001,
    max_hold_bars: int = 60,
    strength_lookback: int = 20,
) -> list[TradeResult]:
    """バックテストを実行する.

    Args:
        m5_data: 5分足データ辞書
        d1_data: 日足データ辞書
        ma_period: MA期間（5分足上、120=1時間足10MA相当）
        pivot_tolerance_pips: ピボット接触判定の許容pips
        sl_buffer_pips: SLのバッファpips
        strength_threshold: 通貨強弱の最低スコア差
        max_hold_bars: 最大保有バー数（5分足）
        strength_lookback: 通貨強弱計算のルックバック期間

    Returns:
        トレード結果のリスト
    """
    # 日足ピボットを全ペアで計算
    pivots = {}
    for pair, d1 in d1_data.items():
        pivots[pair] = calc_daily_pivots(d1)

    # 終値辞書を作成（通貨強弱計算用）
    closes_dict = {}
    for pair, m5 in m5_data.items():
        closes_dict[pair] = m5[["close"]]

    # 通貨強弱を計算
    strength = calc_currency_strength(closes_dict, lookback=strength_lookback)

    if len(strength) == 0:
        print("通貨強弱の計算に失敗")
        return []

    # 共通インデックスを取得
    common_idx = strength.index
    for pair, m5 in m5_data.items():
        common_idx = common_idx.intersection(m5.index)
    common_idx = common_idx.sort_values()

    trades: list[TradeResult] = []
    open_position: dict | None = None
    cooldown_until: pd.Timestamp | None = None

    for i, t in enumerate(common_idx):
        # ポジションがある場合: 決済チェック
        if open_position is not None:
            sig = open_position["signal"]
            pair = sig.pair
            if pair not in m5_data or t not in m5_data[pair].index:
                continue

            bar = m5_data[pair].loc[t]
            hold_bars = open_position["hold_bars"] + 1

            # SLチェック
            hit_sl = False
            hit_tp = False

            if sig.direction == "long":
                if bar["low"] <= sig.sl_price:
                    hit_sl = True
                if bar["high"] >= sig.tp_price:
                    hit_tp = True
            else:
                if bar["high"] >= sig.sl_price:
                    hit_sl = True
                if bar["low"] <= sig.tp_price:
                    hit_tp = True

            if hit_sl:
                pips = _calc_pips(sig.entry_price, sig.sl_price, sig.direction, pair)
                trades.append(TradeResult(
                    signal=sig, exit_time=t, exit_price=sig.sl_price,
                    pips=pips, result="SL", hold_bars=hold_bars,
                ))
                cooldown_until = t + pd.Timedelta(minutes=30)
                open_position = None
                continue
            elif hit_tp:
                pips = _calc_pips(sig.entry_price, sig.tp_price, sig.direction, pair)
                trades.append(TradeResult(
                    signal=sig, exit_time=t, exit_price=sig.tp_price,
                    pips=pips, result="TP", hold_bars=hold_bars,
                ))
                cooldown_until = t + pd.Timedelta(minutes=30)
                open_position = None
                continue
            elif hold_bars >= max_hold_bars:
                exit_price = bar["close"]
                pips = _calc_pips(sig.entry_price, exit_price, sig.direction, pair)
                trades.append(TradeResult(
                    signal=sig, exit_time=t, exit_price=exit_price,
                    pips=pips, result="timeout", hold_bars=hold_bars,
                ))
                cooldown_until = t + pd.Timedelta(minutes=30)
                open_position = None
                continue

            open_position["hold_bars"] = hold_bars
            continue

        # クールダウン中はスキップ
        if cooldown_until is not None and t < cooldown_until:
            continue

        # MA計算に十分なデータがあるかチェック
        if i < ma_period + 10:
            continue

        # --- シグナル判定 ---

        # 1. 通貨強弱を取得
        if t not in strength.index:
            continue
        strength_row = strength.loc[t]
        strongest, weakest, score_diff = find_strongest_weakest(strength_row)

        # 強弱差が閾値以下ならスキップ
        if score_diff < strength_threshold:
            continue

        # 2. トレードペアを特定
        pair = get_pair_for_currencies(strongest, weakest)
        if pair is None or pair not in m5_data:
            continue

        direction = get_direction_for_pair(pair, strongest, weakest)

        # 3. 現在価格を取得
        if t not in m5_data[pair].index:
            continue
        current_bar = m5_data[pair].loc[t]
        price = current_bar["close"]

        # 4. デイリーピボットを取得
        if pair not in pivots:
            continue
        pivot_row = _get_daily_pivot_for_time(pivots[pair], t)
        if pivot_row is None:
            continue

        pivot_p = pivot_row["P"]

        # 5. ピボットP0への接触チェック
        if not is_near_pivot(price, pivot_p, pivot_tolerance_pips, pair):
            continue

        # 6. MA方向フィルター
        closes_up_to = m5_data[pair]["close"].loc[:t]
        if not check_ma_direction(closes_up_to, ma_period, direction):
            continue

        # 7. エントリー
        sl, tp = calc_sl_tp(
            entry_price=price,
            direction=direction,
            pivot_p=pivot_p,
            pivot_s1=pivot_row["S1"],
            pivot_r1=pivot_row["R1"],
            pair=pair,
            sl_buffer_pips=sl_buffer_pips,
        )

        # RR比チェック（最低1.0以上）
        pip_val = _get_pip_value(pair)
        risk = abs(price - sl) / pip_val
        reward = abs(tp - price) / pip_val
        if risk <= 0 or reward / risk < 1.0:
            continue

        signal = OikawaSignal(
            time=t, pair=pair, direction=direction,
            entry_price=price, sl_price=sl, tp_price=tp,
            pivot_p=pivot_p, strongest=strongest, weakest=weakest,
            strength_score=score_diff,
        )

        open_position = {"signal": signal, "hold_bars": 0}

    return trades


def _calc_pips(
    entry: float, exit_price: float, direction: str, pair: str,
) -> float:
    """pips計算."""
    pip_val = _get_pip_value(pair)
    if direction == "long":
        return round((exit_price - entry) / pip_val, 1)
    return round((entry - exit_price) / pip_val, 1)


def calc_stats(trades: list[TradeResult]) -> dict:
    """トレード統計を計算する."""
    if not trades:
        return {"total": 0, "message": "トレードなし"}

    total = len(trades)
    wins = [t for t in trades if t.pips > 0]
    losses = [t for t in trades if t.pips <= 0]

    win_rate = len(wins) / total * 100
    total_pips = sum(t.pips for t in trades)
    avg_pips = total_pips / total

    avg_win = np.mean([t.pips for t in wins]) if wins else 0
    avg_loss = np.mean([t.pips for t in losses]) if losses else 0

    pf = (
        abs(sum(t.pips for t in wins)) / abs(sum(t.pips for t in losses))
        if losses and sum(t.pips for t in losses) != 0
        else float("inf")
    )

    # ペア別集計
    pair_stats = {}
    for t in trades:
        if t.signal.pair not in pair_stats:
            pair_stats[t.signal.pair] = {"trades": 0, "pips": 0, "wins": 0}
        pair_stats[t.signal.pair]["trades"] += 1
        pair_stats[t.signal.pair]["pips"] += t.pips
        if t.pips > 0:
            pair_stats[t.signal.pair]["wins"] += 1

    # 結果タイプ別
    tp_count = sum(1 for t in trades if t.result == "TP")
    sl_count = sum(1 for t in trades if t.result == "SL")
    timeout_count = sum(1 for t in trades if t.result == "timeout")

    # 最大連敗
    max_losing_streak = 0
    current_streak = 0
    for t in trades:
        if t.pips <= 0:
            current_streak += 1
            max_losing_streak = max(max_losing_streak, current_streak)
        else:
            current_streak = 0

    # 最大ドローダウン
    cumulative = np.cumsum([t.pips for t in trades])
    peak = np.maximum.accumulate(cumulative)
    drawdown = peak - cumulative
    max_dd = np.max(drawdown) if len(drawdown) > 0 else 0

    return {
        "total": total,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(win_rate, 1),
        "total_pips": round(total_pips, 1),
        "avg_pips": round(avg_pips, 1),
        "avg_win": round(avg_win, 1),
        "avg_loss": round(avg_loss, 1),
        "profit_factor": round(pf, 2),
        "max_losing_streak": max_losing_streak,
        "max_drawdown": round(max_dd, 1),
        "tp_count": tp_count,
        "sl_count": sl_count,
        "timeout_count": timeout_count,
        "pair_stats": pair_stats,
        "avg_hold_bars": round(np.mean([t.hold_bars for t in trades]), 1),
    }


def trades_to_dataframe(trades: list[TradeResult]) -> pd.DataFrame:
    """トレード結果をDataFrameに変換する."""
    if not trades:
        return pd.DataFrame()

    rows = []
    for t in trades:
        rows.append({
            "entry_time": t.signal.time,
            "exit_time": t.exit_time,
            "pair": t.signal.pair,
            "direction": t.signal.direction,
            "entry_price": t.signal.entry_price,
            "exit_price": t.exit_price,
            "sl": t.signal.sl_price,
            "tp": t.signal.tp_price,
            "pips": t.pips,
            "result": t.result,
            "hold_bars": t.hold_bars,
            "strongest": t.signal.strongest,
            "weakest": t.signal.weakest,
            "strength_score": round(t.signal.strength_score, 6),
            "pivot_p": t.signal.pivot_p,
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("=" * 60)
    print("及川式通貨相関トレード バックテスト")
    print("=" * 60)

    print("\nデータ取得中...")
    m5_data, d1_data = fetch_backtest_data(count=5000)
    print(f"  取得ペア数: M5={len(m5_data)}, D1={len(d1_data)}")

    for pair, df in m5_data.items():
        print(f"  {pair}: {len(df)}本 ({df.index[0]} ~ {df.index[-1]})")

    print("\nバックテスト実行中...")
    trades = run_backtest(m5_data, d1_data)
    stats = calc_stats(trades)

    print(f"\n--- 結果 ---")
    print(f"トレード数: {stats['total']}")

    if stats["total"] > 0:
        print(f"勝率: {stats['win_rate']}%")
        print(f"総Pips: {stats['total_pips']}")
        print(f"平均Pips: {stats['avg_pips']}")
        print(f"PF: {stats['profit_factor']}")
        print(f"平均勝ち: {stats['avg_win']} pips")
        print(f"平均負け: {stats['avg_loss']} pips")
        print(f"最大連敗: {stats['max_losing_streak']}")
        print(f"最大DD: {stats['max_drawdown']} pips")
        print(f"TP: {stats['tp_count']}, SL: {stats['sl_count']}, Timeout: {stats['timeout_count']}")
        print(f"平均保有: {stats['avg_hold_bars']}バー")

        print(f"\n--- ペア別 ---")
        for pair, ps in stats["pair_stats"].items():
            wr = ps["wins"] / ps["trades"] * 100 if ps["trades"] > 0 else 0
            print(f"  {pair}: {ps['trades']}T, WR={wr:.0f}%, Pips={ps['pips']:.1f}")

        # CSV保存
        df = trades_to_dataframe(trades)
        df.to_csv("oikawa_backtest_result.csv", index=False)
        print("\n結果をoikawa_backtest_result.csvに保存しました")
