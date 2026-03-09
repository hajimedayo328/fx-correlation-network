"""経済指標カレンダー データ取得.

Forex Factoryから経済指標の発表スケジュール・結果をスクレイピングし、
通貨ペアの価格データと紐づけるためのモジュール。

データソース: Forex Factory (https://www.forexfactory.com/calendar)
キャッシュ: ローカルCSVに保存して再利用
"""

from __future__ import annotations

import re
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

CACHE_DIR = Path(__file__).parent / "cache"

# 主要経済指標（High Impact のみ対象）
MAJOR_INDICATORS = {
    "USD": [
        "Non-Farm Employment Change",
        "CPI m/m",
        "Core CPI m/m",
        "FOMC Statement",
        "Federal Funds Rate",
        "GDP q/q",
        "Advance GDP q/q",
        "Prelim GDP q/q",
        "Unemployment Rate",
        "Retail Sales m/m",
        "Core Retail Sales m/m",
        "ISM Manufacturing PMI",
        "ISM Services PMI",
        "PPI m/m",
        "Core PPI m/m",
        "ADP Non-Farm Employment Change",
    ],
    "EUR": [
        "Main Refinancing Rate",
        "ECB Press Conference",
        "CPI Flash Estimate y/y",
        "Core CPI Flash Estimate y/y",
        "German Prelim CPI m/m",
        "Flash Manufacturing PMI",
        "Flash Services PMI",
        "GDP Flash Estimate q/q",
    ],
    "GBP": [
        "Official Bank Rate",
        "CPI y/y",
        "GDP m/m",
        "Flash Manufacturing PMI",
        "Flash Services PMI",
        "Claimant Count Change",
    ],
    "JPY": [
        "BOJ Policy Rate",
        "BOJ Press Conference",
        "National Core CPI y/y",
        "Tankan Manufacturing Index",
        "GDP q/q",
    ],
    "AUD": [
        "Cash Rate",
        "RBA Rate Statement",
        "CPI q/q",
        "Employment Change",
        "Unemployment Rate",
        "GDP q/q",
    ],
}

# 指標名 → 短縮名マッピング
INDICATOR_SHORT_NAMES = {
    "Non-Farm Employment Change": "NFP",
    "CPI m/m": "CPI",
    "Core CPI m/m": "Core CPI",
    "FOMC Statement": "FOMC",
    "Federal Funds Rate": "Fed Rate",
    "GDP q/q": "GDP",
    "Advance GDP q/q": "GDP(Adv)",
    "Prelim GDP q/q": "GDP(Pre)",
    "Unemployment Rate": "失業率",
    "Retail Sales m/m": "小売売上",
    "ISM Manufacturing PMI": "ISM製造業",
    "ISM Services PMI": "ISMサービス",
    "ADP Non-Farm Employment Change": "ADP雇用",
    "Main Refinancing Rate": "ECB金利",
    "ECB Press Conference": "ECB会見",
    "Official Bank Rate": "BOE金利",
    "BOJ Policy Rate": "BOJ金利",
    "Cash Rate": "RBA金利",
}


def _parse_forex_factory_page(html: str, year: int) -> list[dict]:
    """Forex FactoryのHTML1ページを解析してイベントリストを返す."""
    soup = BeautifulSoup(html, "html.parser")
    rows = soup.select("tr.calendar__row")

    events = []
    current_date = None

    for row in rows:
        # 日付セル
        date_cell = row.select_one("td.calendar__date span")
        if date_cell and date_cell.text.strip():
            date_text = date_cell.text.strip()
            # "Mon Jan 6" のようなフォーマット
            try:
                current_date = datetime.strptime(
                    f"{date_text} {year}", "%a %b %d %Y"
                ).date()
            except ValueError:
                pass

        if current_date is None:
            continue

        # 時刻
        time_cell = row.select_one("td.calendar__time span")
        time_text = time_cell.text.strip() if time_cell else ""

        # 通貨
        currency_cell = row.select_one("td.calendar__currency")
        currency = currency_cell.text.strip() if currency_cell else ""

        # インパクト
        impact_cell = row.select_one("td.calendar__impact span")
        impact = ""
        if impact_cell:
            cls = impact_cell.get("class", [])
            for c in cls:
                if "high" in c.lower():
                    impact = "High"
                elif "medium" in c.lower():
                    impact = "Medium"
                elif "low" in c.lower():
                    impact = "Low"

        # イベント名
        event_cell = row.select_one("td.calendar__event span")
        event_name = event_cell.text.strip() if event_cell else ""

        # Actual / Forecast / Previous
        actual_cell = row.select_one("td.calendar__actual span")
        forecast_cell = row.select_one("td.calendar__forecast span")
        previous_cell = row.select_one("td.calendar__previous span")

        actual = actual_cell.text.strip() if actual_cell else ""
        forecast = forecast_cell.text.strip() if forecast_cell else ""
        previous = previous_cell.text.strip() if previous_cell else ""

        if not event_name or not currency:
            continue

        # 時刻パース
        event_dt = None
        if time_text and re.match(r"\d{1,2}:\d{2}(am|pm)", time_text):
            try:
                t = datetime.strptime(time_text, "%I:%M%p").time()
                event_dt = datetime.combine(current_date, t)
            except ValueError:
                pass

        events.append({
            "datetime": event_dt,
            "date": current_date,
            "time": time_text,
            "currency": currency,
            "impact": impact,
            "event": event_name,
            "actual": actual,
            "forecast": forecast,
            "previous": previous,
        })

    return events


def fetch_forex_factory_calendar(
    start_date: datetime,
    end_date: datetime,
    impact_filter: str = "High",
) -> pd.DataFrame:
    """Forex Factoryから経済指標カレンダーをスクレイピングする.

    Args:
        start_date: 取得開始日
        end_date: 取得終了日
        impact_filter: "High", "Medium", "Low", "All"

    Returns:
        経済指標イベントのDataFrame
    """
    CACHE_DIR.mkdir(exist_ok=True)
    cache_key = f"ff_calendar_{start_date:%Y%m%d}_{end_date:%Y%m%d}_{impact_filter}"
    cache_path = CACHE_DIR / f"{cache_key}.csv"

    # キャッシュがあれば使う（1日以内なら）
    if cache_path.exists():
        mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
        if datetime.now() - mtime < timedelta(hours=24):
            df = pd.read_csv(cache_path, parse_dates=["datetime", "date"])
            return df

    all_events = []
    current = start_date

    while current <= end_date:
        # Forex Factoryの週URL: ?week=jan6.2025
        week_str = current.strftime("%b").lower() + current.strftime("%-d.%Y")
        # Windows互換
        week_str = current.strftime("%b").lower() + f"{current.day}.{current.year}"
        url = f"https://www.forexfactory.com/calendar?week={week_str}"

        try:
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
                "Accept-Language": "en-US,en;q=0.9",
            }
            resp = requests.get(url, headers=headers, timeout=15)
            if resp.status_code == 200:
                events = _parse_forex_factory_page(resp.text, current.year)
                all_events.extend(events)

            # レート制限対策
            time.sleep(2)

        except Exception as e:
            print(f"Forex Factory取得エラー ({week_str}): {e}")

        # 次の週へ
        current += timedelta(weeks=1)

    if not all_events:
        return pd.DataFrame(
            columns=[
                "datetime", "date", "time", "currency",
                "impact", "event", "actual", "forecast", "previous",
            ]
        )

    df = pd.DataFrame(all_events)

    # インパクトフィルター
    if impact_filter != "All":
        df = df[df["impact"] == impact_filter]

    # キャッシュ保存
    df.to_csv(cache_path, index=False)

    return df.reset_index(drop=True)


def create_builtin_calendar() -> pd.DataFrame:
    """スクレイピング不要のビルトイン経済指標カレンダー.

    主要な定期発表指標の典型的なスケジュールを生成する。
    実際の発表日時と完全に一致するわけではないが、
    分析のデモ・テスト用として使える。
    """
    # 直近1年分の主要イベントを手動定義
    # 実際の運用ではForex Factoryから取得するが、
    # スクレイピングがブロックされた場合のフォールバック
    events = []

    # 米国NFP: 毎月第1金曜 13:30 UTC (22:30 JST)
    # CPI: 毎月中旬 13:30 UTC
    # FOMC: 年8回 19:00 UTC
    # これらの典型的な日付を生成

    base_year = datetime.now().year
    for month in range(1, 13):
        # NFP（第1金曜を近似）
        nfp_date = datetime(base_year, month, 1)
        # 第1金曜を計算
        day_of_week = nfp_date.weekday()  # 0=Mon
        days_to_friday = (4 - day_of_week) % 7
        nfp_date = nfp_date + timedelta(days=days_to_friday)

        events.append({
            "datetime": nfp_date.replace(hour=13, minute=30),
            "date": nfp_date.date(),
            "time": "8:30am",
            "currency": "USD",
            "impact": "High",
            "event": "Non-Farm Employment Change",
            "actual": "",
            "forecast": "",
            "previous": "",
        })

        # CPI（15日前後）
        cpi_date = datetime(base_year, month, 13)
        events.append({
            "datetime": cpi_date.replace(hour=13, minute=30),
            "date": cpi_date.date(),
            "time": "8:30am",
            "currency": "USD",
            "impact": "High",
            "event": "CPI m/m",
            "actual": "",
            "forecast": "",
            "previous": "",
        })

    # FOMC（年8回: 1,3,5,6,7,9,11,12月）
    fomc_months = [1, 3, 5, 6, 7, 9, 11, 12]
    for month in fomc_months:
        fomc_date = datetime(base_year, month, 28 if month != 12 else 18)
        events.append({
            "datetime": fomc_date.replace(hour=19, minute=0),
            "date": fomc_date.date(),
            "time": "2:00pm",
            "currency": "USD",
            "impact": "High",
            "event": "Federal Funds Rate",
            "actual": "",
            "forecast": "",
            "previous": "",
        })

    return pd.DataFrame(events)


def get_indicator_short_name(event_name: str) -> str:
    """指標名の短縮名を返す."""
    return INDICATOR_SHORT_NAMES.get(event_name, event_name[:15])


def filter_calendar_by_currency(
    calendar_df: pd.DataFrame,
    currencies: list[str] | None = None,
) -> pd.DataFrame:
    """通貨でフィルターする."""
    if currencies is None:
        currencies = ["USD", "EUR", "GBP", "JPY", "AUD"]
    return calendar_df[calendar_df["currency"].isin(currencies)].reset_index(drop=True)
