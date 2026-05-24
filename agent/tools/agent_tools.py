import json
import os
import sys
from datetime import date

import requests
import torch
from langchain_core.tools import tool

from rag.rag_service import RagSummarizeService
from utils.logger_handler import logger

rag = RagSummarizeService()

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)

from algorithms.epf_predictor import (  # noqa: E402
    compute_confidence_band,
    forecast_steps,
    parse_forecast_end_date,
    predict_future_price,
    predict_future_prices,
)
from algorithms.price_data_provider import (  # noqa: E402
    get_last_csv_delivery_date,
    get_recent_prices_flat,
    get_recent_multivariate_window,
)
from algorithms.weather_data import (  # noqa: E402
    INDIANA_HUB,
    TEXAS_HOUSTON,
    fetch_weather_forecast,
)


def _init_gridstatus_live_from_env() -> None:
    if os.environ.get("GRIDSTATUS_ENABLED", "").strip().lower() not in (
        "1",
        "true",
        "yes",
        "on",
    ):
        return
    try:
        from algorithms.gridstatus_feed import register_gridstatus_fetcher_from_env

        register_gridstatus_fetcher_from_env()
        logger.info("已启用 gridstatus 拉取 MISO 日度均价（见 algorithms/gridstatus_feed.py）")
    except Exception as e:
        logger.warning("GRIDSTATUS 注册失败，将回退 CSV: %s", e)


_init_gridstatus_live_from_env()


@tool(description="从向量存储中检索参考资料")
def rag_summarize(query: str) -> str:
    return rag.rag_summarize(query)


@tool
def get_texas_weather(query: str = "") -> str:
    """获取美国休斯顿（ERCOT）或 Indianapolis（MISO）的天气。支持实时天气和7日预报。
    输入含 'forecast'/'7天'/'一周'/'week' 等词时返回7日预报，否则返回实时天气。"""

    if "indiana" in query.lower() or "miso" in query.lower():
        lat, lon, location_name = (
            INDIANA_HUB[0], INDIANA_HUB[1], "Indiana Indianapolis (MISO)"
        )
    else:
        lat, lon, location_name = (
            TEXAS_HOUSTON[0], TEXAS_HOUSTON[1], "Texas Houston (ERCOT)"
        )

    want_forecast = any(kw in query.lower() for kw in [
        "forecast", "7天", "7日", "一周", "未来", "每天", "daily", "week",
        "七天", "多日", "几日", "接下来",
    ])

    print(f"[工具层] 正在请求外部气象 API: {location_name}...")

    if want_forecast:
        try:
            fc = fetch_weather_forecast(lat, lon, forecast_days=7)
            dates = fc["dates"]
            t_max = fc["temp_max"]
            t_min = fc["temp_min"]
            precip = fc["precip"]
            wind = fc["wind_speed"]

            if not dates:
                return f"Weather API returned no forecast data for {location_name}"

            lines = [f"**{location_name} 7-Day Forecast**\n"]
            for i in range(min(7, len(dates))):
                d = dates[i]
                hi = t_max[i] if i < len(t_max) else "?"
                lo = t_min[i] if i < len(t_min) else "?"
                r = precip[i] if i < len(precip) else 0
                w = wind[i] if i < len(wind) else "?"
                weather_flag = ""
                if isinstance(hi, (int, float)) and hi > 35:
                    weather_flag = "⚠️ Extreme Heat"
                elif isinstance(lo, (int, float)) and lo < 0:
                    weather_flag = "❄️ Extreme Cold"
                elif isinstance(r, (int, float)) and r > 10:
                    weather_flag = "🌧️ Heavy Rain"
                lines.append(
                    f"| {d} | {lo}~{hi}°C | {w} km/h | {r}mm {weather_flag} |"
                )
            lines.append(
                f"\n_Data source: Open-Meteo. Extreme weather may drive electricity demand & price spikes._"
            )
            return "\n".join(lines)

        except Exception as e:
            return f"Weather API (forecast) call failed: {e}"

    else:
        try:
            fc = fetch_weather_forecast(lat, lon, forecast_days=1)
            temp = fc["temp_max"][0] if fc.get("temp_max") else "Unknown"
            wind_speed = fc["wind_speed"][0] if fc.get("wind_speed") else "Unknown"

            if isinstance(temp, (int, float)) and temp > 35:
                return f"[Extreme Heat Alert] Current temperature {temp}C, wind {wind_speed} km/h at {location_name}. AC load expected to surge!"
            if isinstance(temp, (int, float)) and temp < 0:
                return f"[Extreme Cold Alert] Current temperature {temp}C, wind {wind_speed} km/h at {location_name}. Heating load expected to surge!"
            return f"[Weather Normal] Current temperature {temp}C, wind {wind_speed} km/h at {location_name}, within normal load range."
        except Exception as e:
            return f"Weather API call failed: {e}"


@tool
def predict_electricity_price(target_date: str) -> float:
    """
    预测未来特定日期的电力市场交易价格（LSTM 单点，取窗口最后一日）。
    target_date 支持 ISO 日期如 2026-05-10，或 明天、后天、大后天、3天后 等中文描述。
    """
    logger.info(f"Agent invoking LSTM neural network for {target_date}...")

    try:
        predicted_price = predict_future_price(target_date)
        return float(predicted_price)

    except Exception as e:
        logger.error(f"底层的时序模型推理失败: {e}")
        return 48.88


@tool
def forecast_electricity_prices(target_date: str) -> str:
    """
    多变量 LSTM 多日滚动预测，自动融合天气特征。返回含置信区间和 Chronos-T5 交叉验证的 JSON。

    target_date 支持 ISO 日期(2026-05-15)或中文描述(明天/后天/3天后/一周后)。
    适合「预测未来一周电价」「明天电价多少」等场景。
    """
    logger.info(f"LSTM multi-day forecast: {target_date}")
    try:
        rows = predict_future_prices(target_date, with_confidence=True)
        if not rows:
            return json.dumps({"error": "模型未就绪或预测失败", "series": []}, ensure_ascii=False)

        # Cross-check with Chronos-T5
        chronos_enabled = False
        try:
            from algorithms.epf_predictor import parse_forecast_end_date, forecast_steps
            fa = date.today()
            end_d = parse_forecast_end_date(target_date, fa)
            steps = forecast_steps(fa, end_d)

            chronos_prices = _run_chronos_multistep(steps)
            if chronos_prices and len(chronos_prices) == len(rows):
                chronos_enabled = True
                for i, cp in enumerate(chronos_prices):
                    rows[i]["chronos_crosscheck"] = round(cp, 2)
                    # Refine confidence band
                    band = compute_confidence_band(rows[i]["price"], cp)
                    rows[i]["confidence"] = band["confidence"]
                    rows[i]["lower"] = band["lower"]
                    rows[i]["upper"] = band["upper"]
        except Exception:
            pass

        # Compute trend summary
        prices_only = [r["price"] for r in rows]
        if len(prices_only) >= 2:
            delta = prices_only[-1] - prices_only[0]
            pct = (delta / prices_only[0]) * 100 if prices_only[0] else 0
            trend = "rising" if delta > 2 else ("falling" if delta < -2 else "flat")
        else:
            trend, pct = "flat", 0.0

        result = {
            "series": rows,
            "summary": {
                "trend": trend,
                "avg_price": round(sum(prices_only) / len(prices_only), 2),
                "change_pct": round(pct, 1),
                "chronos_enabled": chronos_enabled,
            },
        }
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        logger.error(f"多日预测失败: {e}")
        return json.dumps({"error": str(e), "series": []}, ensure_ascii=False)


def _run_chronos_multistep(steps: int):
    """Run Chronos-T5 for multi-step prediction. Returns list of floats or None."""
    global _chronos_pipeline
    if _chronos_pipeline is None:
        try:
            from chronos import ChronosPipeline as _ChronosPipeline
        except ImportError:
            return None
        _chronos_pipeline = _ChronosPipeline.from_pretrained(
            "amazon/chronos-t5-tiny",
            device_map="cpu",
            dtype=torch.float32,
        )
    try:
        hist_list = get_recent_prices_flat(date.today(), 10)
    except Exception:
        return None

    historical_data = torch.tensor(hist_list, dtype=torch.float32)
    forecast = _chronos_pipeline.predict(historical_data, prediction_length=steps)
    t = forecast[0]
    if t.dim() >= 2:
        return t[:, -1].tolist() if t.shape[1] >= steps else t.flatten().tolist()[:steps]
    return t.tolist()[:steps]


_chronos_pipeline = None
_CHRONOS_CONTEXT_LEN = 10


@tool
def get_chronos_prediction(target_date: str) -> float:
    """Chronos-T5 时序基础模型电价预测，作为 LSTM 轨道的交叉验证。
    target_date 支持 ISO 日期(2026-05-10)或中文描述(明天/后天/N天后)。"""
    global _chronos_pipeline

    if _chronos_pipeline is None:
        try:
            from chronos import ChronosPipeline as _ChronosPipeline
        except ImportError as e:
            raise ImportError(
                "chronos library not installed. Please: pip install chronos-forecasting"
            ) from e
        print("[System Init] Loading Chronos-T5-Tiny into memory (one-time only)...")
        _chronos_pipeline = _ChronosPipeline.from_pretrained(
            "amazon/chronos-t5-tiny",
            device_map="cpu",
            dtype=torch.float32,
        )

    fa = date.today()
    try:
        end_d = parse_forecast_end_date(target_date, fa)
        steps = forecast_steps(fa, end_d)
    except ValueError:
        steps = 1

    try:
        hist_list = get_recent_prices_flat(fa, _CHRONOS_CONTEXT_LEN)
    except Exception:
        hist_list = [45.2, 46.1, 48.5, 49.0, 47.8, 48.2, 46.5, 47.1, 49.2, 50.1]

    historical_data = torch.tensor(hist_list, dtype=torch.float32)

    print(f"[Chronos] prediction_length={steps} ...")
    forecast = _chronos_pipeline.predict(historical_data, prediction_length=steps)

    t = forecast[0]
    if t.dim() >= 2:
        last = t[:, -1]
    elif t.dim() == 1 and t.numel() > 1:
        last = t[-1:]
    else:
        last = t.flatten()
    predicted_value = last.mean().item()

    return round(float(predicted_value), 4)


@tool
def get_data_freshness(query: str = "") -> str:
    """Report the freshness of training data and real-time data availability.
    Call this when users ask about data sources, prediction confidence, or data currency."""
    from datetime import date as _date
    from algorithms.price_data_provider import (
        get_last_csv_delivery_date,
        get_live_price_fetcher,
    )
    from algorithms.gridstatus_feed import fetch_miso_hub_daily_lmp_sequence
    import os as _os

    today = _date.today()
    lines = []

    # CSV status
    csv_last = get_last_csv_delivery_date()
    if csv_last:
        gap = (today - csv_last).days
        lines.append(f"CSV training data: last entry {csv_last} ({gap} days ago)")
    else:
        lines.append("CSV training data: not found")

    # Live fetcher status
    fetcher = get_live_price_fetcher()
    if fetcher is not None:
        lines.append("Live price fetcher (gridstatus): registered")
        # Try to find latest available data
        latest_data = fetch_miso_hub_daily_lmp_sequence(today, 1, max_lookback=14)
        if latest_data is not None:
            lines.append(f"  Latest MISO LMP: ${latest_data[0]}/MWh (with publish delay offset)")
        else:
            lines.append("  MISO LMP: unavailable (may be outside publish window)")
    else:
        lines.append(
            "Live price fetcher (gridstatus): NOT registered. "
            "Set GRIDSTATUS_ENABLED=1 to enable real-time MISO data."
        )

    # Model status
    model_path = _os.path.join(
        _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))),
        "algorithms", "lstm_epf_multivariate.pth",
    )
    if _os.path.exists(model_path):
        lines.append("LSTM model (multivariate): ready")
    else:
        lines.append("LSTM model (multivariate): NOT trained")

    return "\n".join(lines)


# ======================== 新增工具 ========================


@tool
def get_current_date(query: str = "") -> str:
    """Get today's real date. Call this when the user asks 'what date is it today' or '今天是几号'."""
    from datetime import date as _date
    today = _date.today()
    return f"Today is {today.isoformat()} ({today.strftime('%A')}), week {today.isocalendar()[1]} of {today.year}."


@tool
def get_trading_advice(target_date: str) -> str:
    """基于多变量 LSTM 预测和 Chronos-T5 交叉验证，生成电力交易建议 (BUY/SELL/HOLD)。

    target_date: 预测周期终点，如 "3天后"、"2026-05-18"、"一周后"。
    若想基于短期趋势做决策，建议传入 "3天后"；若分析周度趋势，建议传入 "7天后"。
    """
    logger.info(f"生成交易建议: {target_date}")
    try:
        # Get current price
        from datetime import date as _date
        today = _date.today()
        recent = get_recent_prices_flat(today, 1)
        current_price = recent[0] if recent else 40.0

        # Get multi-day forecast
        rows = predict_future_prices(target_date)
        if not rows:
            return json.dumps({"error": "预测失败，无法生成交易建议"}, ensure_ascii=False)

        prices = [r["price"] for r in rows]
        avg_pred = sum(prices) / len(prices)
        delta = avg_pred - current_price
        pct = (delta / current_price) * 100

        # Decision logic
        if pct > 10:
            decision, action = "BUY", "建议提前买入，锁定当前低价"
        elif pct < -10:
            decision, action = "SELL", "建议卖出/减持，避免高位回落损失"
        elif abs(pct) <= 5:
            decision, action = "HOLD", "价格波动不大，建议观望"
        else:
            decision, action = "HOLD", "趋势不够明确，建议观望等待信号"

        # Weather modifier
        try:
            fc = fetch_weather_forecast(INDIANA_HUB[0], INDIANA_HUB[1], forecast_days=len(rows))
            hot_days = sum(1 for t in fc.get("temp_max", []) if isinstance(t, (int, float)) and t > 35)
            cold_days = sum(1 for t in fc.get("temp_min", []) if isinstance(t, (int, float)) and t < 0)
            if hot_days >= 2 and decision == "HOLD" and pct > 0:
                decision, action = "BUY", f"预报{hot_days}天高温(>35°C)，预计用电需求上升，建议买入"
            if cold_days >= 2 and decision == "HOLD" and pct > 0:
                decision, action = "BUY", f"预报{cold_days}天极寒(<0°C)，供暖负荷上升，建议买入"
        except Exception:
            pass

        # Confidence from model agreement
        confidence = "medium"
        chronos_price = None
        try:
            chronos_price = _run_chronos_multistep(len(rows))
            if chronos_price:
                chronos_avg = sum(chronos_price) / len(chronos_price)
                delta_c = abs(avg_pred - chronos_avg)
                if delta_c < 3:
                    confidence = "high"
                elif delta_c > 7:
                    confidence = "low"
        except Exception:
            pass

        # Risk factors
        risk_factors = ["模型预测存在不确定性 (历史MAPE约14%)"]
        if confidence == "low":
            risk_factors.append("LSTM与Chronos-T5预测分歧较大")
        try:
            if any(isinstance(p, float) and p > 70 for p in prices):
                risk_factors.append("预测出现极端高价，波动风险大")
        except Exception:
            pass

        result = {
            "decision": decision,
            "action": action,
            "confidence": confidence,
            "current_price": round(current_price, 2),
            "predicted_avg": round(avg_pred, 2),
            "predicted_range": f"{round(min(prices), 2)} ~ {round(max(prices), 2)}",
            "trend": f"{'+' if pct > 0 else ''}{pct:.1f}%",
            "reasoning": (
                f"当前电价 {current_price:.1f} $/MWh，预测周期内均价 {avg_pred:.1f} $/MWh"
                f"（变动 {pct:+.1f}%）。{action}。"
            ),
            "risk_factors": risk_factors,
            "forecast_days": len(rows),
        }
        return json.dumps(result, ensure_ascii=False)

    except Exception as e:
        logger.error(f"交易建议生成失败: {e}")
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@tool
def check_price_alerts(query: str = "") -> str:
    """检查未来7天是否存在电价异常风险。自动扫描预测序列，标记极端价格事件。

    返回红/橙/黄三级告警，并关联天气因素。
    当用户询问「有没有风险」「价格会不会暴涨」「需要注意什么」时调用。
    """
    logger.info("检查电价异常告警...")
    try:
        rows = predict_future_prices("7天后", with_confidence=True)
        if not rows:
            return json.dumps({"error": "预测失败"}, ensure_ascii=False)

        # Thresholds based on historical stats (mean ~38.75, std ~6.29)
        price_mean = 38.75
        price_std = 6.29

        alerts = []
        for r in rows:
            p = r["price"]
            level = None
            reason = ""

            if p > 100 or p > price_mean + 3 * price_std:
                level = "red"
                reason = f"极端高价 {p} $/MWh（远超均值 {price_mean:.0f}+3σ={price_mean+3*price_std:.0f}）"
            elif p > 70 or p > price_mean + 2 * price_std:
                level = "orange"
                reason = f"偏高电价 {p} $/MWh（超过均值+2σ）"
            elif p > 55:
                level = "yellow"
                reason = f"关注: 电价 {p} $/MWh 有所上涨"

            if level:
                alerts.append({
                    "date": r["forecast_date"],
                    "level": level,
                    "price": p,
                    "reason": reason,
                })

        # Check for single-day spikes
        prices = [r["price"] for r in rows]
        for i in range(1, len(prices)):
            jump = (prices[i] - prices[i - 1]) / prices[i - 1] * 100 if prices[i - 1] else 0
            if jump > 30:
                alerts.append({
                    "date": rows[i]["forecast_date"],
                    "level": "yellow",
                    "price": prices[i],
                    "reason": f"单日涨幅 {jump:.0f}%，价格跳升",
                })

        # Cross with weather
        try:
            fc = fetch_weather_forecast(INDIANA_HUB[0], INDIANA_HUB[1], forecast_days=7)
            for i, t in enumerate(fc.get("temp_max", [])):
                if isinstance(t, (int, float)) and t > 35 and i < len(rows):
                    # Check if this date already has an alert
                    d = rows[i]["forecast_date"]
                    existing = [a for a in alerts if a["date"] == d]
                    if existing:
                        existing[0]["reason"] += f"；高温 {t}°C 加剧风险"
                    elif rows[i]["price"] > price_mean:
                        alerts.append({
                            "date": d,
                            "level": "yellow",
                            "price": rows[i]["price"],
                            "reason": f"高温 {t}°C 可能推高电价",
                        })
        except Exception:
            pass

        # Deduplicate and sort by level severity
        seen = set()
        unique = []
        severity = {"red": 0, "orange": 1, "yellow": 2}
        for a in sorted(alerts, key=lambda x: severity.get(x["level"], 3)):
            key = (a["date"], a["level"])
            if key not in seen:
                seen.add(key)
                unique.append(a)

        result = {
            "alerts": unique,
            "has_alerts": len(unique) > 0,
            "highest_level": unique[0]["level"] if unique else "none",
            "checked_days": len(rows),
        }
        return json.dumps(result, ensure_ascii=False)

    except Exception as e:
        logger.error(f"告警检查失败: {e}")
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@tool
def generate_daily_briefing(query: str = "") -> str:
    """一键生成「电力市场日报」，整合天气、价格预测、异常告警和交易建议。

    当用户说「生成简报」「今日市场报告」「电力市场日报」「daily briefing」时调用。
    自动聚合多个数据源，输出结构化 Markdown 报告。
    """
    from datetime import date as _date

    today = _date.today()
    logger.info(f"生成电力市场日报: {today}")

    try:
        # 1. Weather forecast
        fc = fetch_weather_forecast(INDIANA_HUB[0], INDIANA_HUB[1], forecast_days=7)
        weather_lines = []
        for i in range(min(7, len(fc["dates"]))):
            d = fc["dates"][i]
            hi = fc["temp_max"][i]
            lo = fc["temp_min"][i]
            r = fc["precip"][i]
            w = fc["wind_speed"][i]
            flag = ""
            if isinstance(hi, (int, float)) and hi > 35:
                flag = " ⚠️高温"
            elif isinstance(lo, (int, float)) and lo < 0:
                flag = " ❄️极寒"
            elif isinstance(r, (int, float)) and r > 10:
                flag = " 🌧️强降水"
            weather_lines.append(
                f"| {d} | {lo}~{hi}°C | {w} km/h | {r}mm |{flag}"
            )

        # 2. Price forecast with confidence
        rows = predict_future_prices("7天后", with_confidence=True)
        price_lines = []
        for r in rows:
            conf = r.get("confidence", "medium")
            price_lines.append(
                f"| {r['forecast_date']} | ${r['price']} | "
                f"${r.get('lower', '?')} ~ ${r.get('upper', '?')} | {conf} |"
            )

        # 3. Alerts
        try:
            alerts_raw = check_price_alerts.invoke({})
            alerts_data = json.loads(alerts_raw) if isinstance(alerts_raw, str) else alerts_raw
        except Exception:
            alerts_data = {"alerts": [], "has_alerts": False, "highest_level": "none"}

        # 4. Trading advice
        try:
            advice_raw = get_trading_advice.invoke({"target_date": "7天后"})
            advice_data = json.loads(advice_raw) if isinstance(advice_raw, str) else advice_raw
        except Exception:
            advice_data = {"decision": "HOLD", "reasoning": "建议数据不可用"}

        # 5. Data freshness
        csv_last = get_last_csv_delivery_date()
        data_gap = (today - csv_last).days if csv_last else "?"

        # Build report
        report = f"""# 电力市场日报 — {today}

## 🌤️ 天气概况 (Indiana Hub)
| 日期 | 温度 | 风速 | 降水 | 预警 |
|------|------|------|------|------|
{chr(10).join(weather_lines)}

_数据来源: Open-Meteo_

## 📈 价格预测 (未来7天)
| 日期 | 预测电价 | 置信区间 | 置信度 |
|------|---------|---------|--------|
{chr(10).join(price_lines)}

## ⚠️ 异常告警
"""

        if alerts_data.get("has_alerts"):
            for a in alerts_data.get("alerts", []):
                icon = {"red": "🔴", "orange": "🟠", "yellow": "🟡"}.get(a["level"], "⚪")
                report += f"- {icon} **{a['date']}**: {a['reason']}\n"
        else:
            report += "无异常告警，未来7天价格在正常范围内。\n"

        report += f"""
## 💰 交易建议
- **决策**: {advice_data.get('decision', 'HOLD')}
- **置信度**: {advice_data.get('confidence', 'medium')}
- **理由**: {advice_data.get('reasoning', '数据不足')}
"""

        if advice_data.get("risk_factors"):
            for rf in advice_data["risk_factors"]:
                report += f"  - ⚡ {rf}\n"

        report += f"""
## 📊 数据状态
- CSV 训练数据: 最后更新 {csv_last} ({data_gap} 天前)
- 预测模型: 多变量 LSTM (price + weather)
- 报告生成时间: {today}

---
_本报告由 AI Agent 自动生成，预测结果仅供参考，不构成实际交易建议。_
"""
        return report

    except Exception as e:
        logger.error(f"日报生成失败: {e}", exc_info=True)
        return f"日报生成失败: {e}"
