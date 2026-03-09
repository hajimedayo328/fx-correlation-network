"""DCC-GARCH動的相関推定.

ローリング相関より精度の高い時変相関を推定する。
fat-tail（裾の重い分布）に強く、危機検出精度が高い。

参考: Musmeci et al. (2017)
"Dynamic correlation network analysis of financial asset returns"
Applied Network Science

手順:
1. 各通貨ペアのリターンにGARCH(1,1)をフィット → 標準化残差を取得
2. 標準化残差間のDCC（動的条件付き相関）を推定
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

try:
    from arch import arch_model
    HAS_ARCH = True
except ImportError:
    HAS_ARCH = False


def _fit_garch_residuals(returns: pd.Series) -> pd.Series | None:
    """GARCH(1,1)をフィットして標準化残差を返す.

    Args:
        returns: 単一通貨ペアのリターン系列

    Returns:
        標準化残差。フィット失敗時はNone
    """
    if not HAS_ARCH:
        return None

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = arch_model(
                returns * 100,  # パーセント表記に変換（数値安定性）
                vol="Garch",
                p=1,
                q=1,
                dist="normal",
                rescale=False,
            )
            result = model.fit(disp="off", show_warning=False)
            std_resid = result.std_resid
            return std_resid
    except Exception:
        return None


def calc_dcc_correlation(
    closes: pd.DataFrame,
    window: int = 50,
) -> dict[str, pd.Series]:
    """DCC-GARCHベースの動的相関を計算する.

    簡易版DCC: GARCH標準化残差のローリング相関を計算。
    完全なDCCモデルではないが、GARCH効果を反映した
    ローリング相関より精度の高い推定。

    Args:
        closes: 終値DataFrame
        window: ローリングウィンドウ

    Returns:
        {"EURUSD / GBPUSD": pd.Series, ...}
    """
    if not HAS_ARCH:
        return {}

    returns = closes.pct_change().dropna()
    symbols = returns.columns.tolist()

    # 各通貨ペアのGARCH標準化残差を計算
    std_resids = {}
    for symbol in symbols:
        resid = _fit_garch_residuals(returns[symbol])
        if resid is not None:
            std_resids[symbol] = resid

    if len(std_resids) < 2:
        return {}

    # 標準化残差間のローリング相関 = DCC近似
    dcc_corrs = {}
    available = list(std_resids.keys())
    for i, src in enumerate(available):
        for j, tgt in enumerate(available):
            if i >= j:
                continue
            key = f"{src} / {tgt}"
            dcc_corrs[key] = (
                std_resids[src]
                .rolling(window)
                .corr(std_resids[tgt])
            )

    return dcc_corrs


def compare_correlation_methods(
    closes: pd.DataFrame,
    pair1: str,
    pair2: str,
    window: int = 50,
) -> pd.DataFrame:
    """ローリング相関とDCC相関を比較する.

    Args:
        closes: 終値DataFrame
        pair1: 通貨ペア1
        pair2: 通貨ペア2
        window: ローリングウィンドウ

    Returns:
        ローリング相関とDCC相関の比較DataFrame
    """
    returns = closes.pct_change().dropna()

    # ローリング相関
    rolling_corr = returns[pair1].rolling(window).corr(returns[pair2])

    result = pd.DataFrame({
        "rolling": rolling_corr,
    })

    # DCC相関
    if HAS_ARCH:
        resid1 = _fit_garch_residuals(returns[pair1])
        resid2 = _fit_garch_residuals(returns[pair2])
        if resid1 is not None and resid2 is not None:
            result["dcc"] = resid1.rolling(window).corr(resid2)

    return result.dropna()
