"""レジーム自動分類.

ネットワーク指標（NTL・密度・Jaccard・代数的連結性）を
組み合わせて市場レジームを自動判定する。

参考:
- Frontiers in Physics (2022): 密度・クラスタリング係数で危機予測
- Onnela et al. (2003): NTL収縮 = 危機
- MDPI Entropy (2021): Jaccard急落 = レジーム転換
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# レジーム定義
REGIME_CRISIS = "危機"
REGIME_TREND = "トレンド"
REGIME_RANGE = "レンジ"
REGIME_TRANSITION = "転換期"


def classify_regime(
    ntl: float,
    density: float,
    jaccard: float,
    ntl_mean: float,
    ntl_std: float,
    density_mean: float,
    density_std: float,
) -> str:
    """単一時点のレジームを分類する.

    ルールベースの分類:
    - 危機: NTLが平均-1σ以下 かつ 密度が平均+1σ以上（全通貨連動・木が収縮）
    - トレンド: 密度が平均+0.5σ以上（通貨間の相関が強い）
    - 転換期: Jaccardが0.5以下（MST構造が急変）
    - レンジ: それ以外

    Args:
        ntl: 現在のNTL
        density: 現在のグラフ密度
        jaccard: 現在のJaccard指数
        ntl_mean: NTLの平均
        ntl_std: NTLの標準偏差
        density_mean: 密度の平均
        density_std: 密度の標準偏差

    Returns:
        レジーム文字列
    """
    ntl_z = (ntl - ntl_mean) / ntl_std if ntl_std > 0 else 0
    density_z = (density - density_mean) / density_std if density_std > 0 else 0

    # 危機: NTL大幅低下 + 密度大幅上昇
    if ntl_z <= -1.0 and density_z >= 1.0:
        return REGIME_CRISIS

    # 転換期: MST構造が急変
    if jaccard < 0.5:
        return REGIME_TRANSITION

    # トレンド: 密度が高い
    if density_z >= 0.5:
        return REGIME_TREND

    return REGIME_RANGE


def classify_rolling_regimes(metrics_df: pd.DataFrame) -> pd.Series:
    """ローリング指標DataFrameからレジーム時系列を生成する.

    Args:
        metrics_df: calc_rolling_all_metricsの返り値
            columns: NTL, Jaccard, density, ...

    Returns:
        時刻→レジーム名 のSeries
    """
    ntl_mean = metrics_df["NTL"].mean()
    ntl_std = metrics_df["NTL"].std()
    density_mean = metrics_df["density"].mean()
    density_std = metrics_df["density"].std()

    regimes = []
    for _, row in metrics_df.iterrows():
        regime = classify_regime(
            ntl=row["NTL"],
            density=row["density"],
            jaccard=row["Jaccard"],
            ntl_mean=ntl_mean,
            ntl_std=ntl_std,
            density_mean=density_mean,
            density_std=density_std,
        )
        regimes.append(regime)

    return pd.Series(regimes, index=metrics_df.index, name="regime")


REGIME_COLORS = {
    REGIME_CRISIS: "#E83929",
    REGIME_TREND: "#5B8930",
    REGIME_RANGE: "#2B3C5E",
    REGIME_TRANSITION: "#E6B422",
}


def get_regime_summary(regimes: pd.Series) -> dict:
    """レジーム分布のサマリーを返す."""
    counts = regimes.value_counts()
    total = len(regimes)
    return {
        regime: f"{count}回 ({count / total * 100:.1f}%)"
        for regime, count in counts.items()
    }
