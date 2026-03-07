"""複数通貨ペアのリアルタイムデータ取得.

MT5から複数通貨ペアの終値を取得し、1つのDataFrameに統合する。
既存の backtest/data/fetcher.py の接続機能を再利用。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# 既存のfetcherをimportできるようにパスを追加
_BACKTEST_DIR = Path(__file__).resolve().parent.parent / "backtest"
if str(_BACKTEST_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKTEST_DIR))

from data.fetcher import connect_mt5, disconnect_mt5, fetch_ohlcv_mt5

# MT5時間足定数
TIMEFRAMES = {
    "M5": 5,
    "M15": 15,
    "M30": 30,
    "H1": 16385,
    "H4": 16388,
    "D1": 16408,
}

# 対象通貨ペア（XAUUSD + EUR/GBP/USD/AUD/JPYの全組み合わせ）
DEFAULT_SYMBOLS = [
    "XAUUSD",
    # ドルストレート
    "EURUSD",
    "GBPUSD",
    "AUDUSD",
    "USDJPY",
    # クロス円
    "EURJPY",
    "GBPJPY",
    "AUDJPY",
    # クロスペア
    "EURGBP",
    "EURAUD",
    "GBPAUD",
]


def fetch_multi_pair_closes(
    symbols: list[str] = DEFAULT_SYMBOLS,
    timeframe_key: str = "H1",
    count: int = 100,
) -> pd.DataFrame:
    """複数通貨ペアの終値を1つのDataFrameに統合する.

    Args:
        symbols: 通貨ペアのリスト
        timeframe_key: 時間足キー ("M5", "M15", "H1", "H4", "D1")
        count: 取得本数

    Returns:
        index=time, columns=通貨ペア名 のDataFrame（終値）
    """
    tf = TIMEFRAMES.get(timeframe_key)
    if tf is None:
        raise ValueError(f"不正な時間足: {timeframe_key}. 選択肢: {list(TIMEFRAMES)}")

    if not connect_mt5():
        raise RuntimeError("MT5接続に失敗しました。MT5を起動してください。")

    try:
        closes = {}
        failed = []

        for symbol in symbols:
            try:
                df = fetch_ohlcv_mt5(symbol, tf, count)
                closes[symbol] = df.set_index("time")["close"]
            except Exception as e:
                failed.append(f"{symbol}: {e}")

        if failed:
            print(f"取得失敗: {', '.join(failed)}")

        if not closes:
            raise RuntimeError("全通貨ペアのデータ取得に失敗しました")

        result = pd.DataFrame(closes)
        result.index.name = "time"
        return result

    finally:
        disconnect_mt5()
