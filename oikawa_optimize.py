"""及川式戦略パラメータ最適化.

主要パラメータを網羅的にテストして最良の組み合わせを探す。
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import pandas as pd

from oikawa_backtest import calc_stats, fetch_backtest_data, run_backtest

# パラメータグリッド
PARAM_GRID = {
    "ma_period": [60, 120, 180],          # 30min / 1H / 1.5H相当
    "pivot_tolerance_pips": [10, 15, 25],  # ピボット接触判定
    "sl_buffer_pips": [3, 5, 10],          # SLバッファ
    "strength_threshold": [0.0005, 0.001, 0.002],  # 通貨強弱閾値
    "max_hold_bars": [36, 60, 96],         # 3H / 5H / 8H
    "strength_lookback": [12, 20, 36],     # 1H / 1.7H / 3H相当
}


def main():
    print("=" * 60)
    print("及川式 パラメータ最適化")
    print("=" * 60)

    # データ取得（1回だけ）
    print("\nデータ取得中...")
    m5_data, d1_data = fetch_backtest_data(count=10000)
    print(f"取得完了: {len(m5_data)}ペア")
    for pair, df in m5_data.items():
        print(f"  {pair}: {len(df)}本 ({df.index[0]} ~ {df.index[-1]})")

    # グリッド生成
    keys = list(PARAM_GRID.keys())
    values = list(PARAM_GRID.values())
    combinations = list(itertools.product(*values))
    total = len(combinations)
    print(f"\nテストパターン数: {total}")

    results = []

    for i, combo in enumerate(combinations):
        params = dict(zip(keys, combo))

        if (i + 1) % 50 == 0 or i == 0:
            print(f"  [{i+1}/{total}] テスト中...")

        trades = run_backtest(m5_data, d1_data, **params)
        stats = calc_stats(trades)

        results.append({
            **params,
            "trades": stats["total"],
            "win_rate": stats.get("win_rate", 0),
            "total_pips": stats.get("total_pips", 0),
            "avg_pips": stats.get("avg_pips", 0),
            "pf": stats.get("profit_factor", 0),
            "max_dd": stats.get("max_drawdown", 0),
            "max_losing": stats.get("max_losing_streak", 0),
            "tp_count": stats.get("tp_count", 0),
            "sl_count": stats.get("sl_count", 0),
            "timeout_count": stats.get("timeout_count", 0),
        })

    df = pd.DataFrame(results)

    # フィルター: 最低10トレード
    df_valid = df[df["trades"] >= 10].copy()

    if len(df_valid) == 0:
        print("\n10トレード以上の組み合わせがありません")
        df.to_csv("oikawa_optimize_all.csv", index=False)
        print("全結果を oikawa_optimize_all.csv に保存")
        return

    # 総合スコア: total_pips重視 + PF + 勝率
    df_valid["score"] = (
        df_valid["total_pips"] * 0.5
        + df_valid["pf"] * 20
        + df_valid["win_rate"] * 0.5
        - df_valid["max_dd"] * 0.1
    )

    df_valid = df_valid.sort_values("score", ascending=False)

    print("\n" + "=" * 60)
    print("TOP 10 パラメータ組み合わせ")
    print("=" * 60)

    for rank, (_, row) in enumerate(df_valid.head(10).iterrows(), 1):
        print(f"\n--- #{rank} (Score: {row['score']:.1f}) ---")
        print(f"  MA={int(row['ma_period'])}, "
              f"Pivot={row['pivot_tolerance_pips']}pips, "
              f"SL_buf={row['sl_buffer_pips']}pips")
        print(f"  Strength={row['strength_threshold']:.4f}, "
              f"MaxHold={int(row['max_hold_bars'])}, "
              f"Lookback={int(row['strength_lookback'])}")
        print(f"  T={int(row['trades'])}, "
              f"WR={row['win_rate']:.1f}%, "
              f"Pips={row['total_pips']:.1f}, "
              f"PF={row['pf']:.2f}, "
              f"DD={row['max_dd']:.1f}")
        print(f"  TP={int(row['tp_count'])}, "
              f"SL={int(row['sl_count'])}, "
              f"Timeout={int(row['timeout_count'])}")

    # CSV保存
    df_valid.to_csv("oikawa_optimize_top.csv", index=False)
    df.to_csv("oikawa_optimize_all.csv", index=False)
    print(f"\n結果保存: oikawa_optimize_top.csv ({len(df_valid)}件)")
    print(f"全結果: oikawa_optimize_all.csv ({len(df)}件)")


if __name__ == "__main__":
    main()
