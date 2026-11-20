"""Offline quantile forecasts for the make.fun GTA VI trophy cases."""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))
LEVELS = (0.05, 0.25, 0.50, 0.75, 0.95)
Z = (-1.645, -0.674, 0.0, 0.674, 1.645)
BASELINE_SIGMA = 0.5
RATE_FLOOR = 0.01
RATE_CEILING = 99.99


def clamp_rate(value):
    return min(RATE_CEILING, max(RATE_FLOOR, float(value)))


def logit_rate(value):
    rate = clamp_rate(value)
    return math.log(rate / (100.0 - rate))


def logistic(value):
    if value >= 0:
        return 100.0 / (1.0 + math.exp(-value))
    e = math.exp(value)
    return 100.0 * e / (1.0 + e)


def parse_time(value):
    stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return stamp.timestamp()


def rounded_rate(value):
    result = round(min(100.0, max(0.0, value)) + 1e-12, 4)
    return int(result) if result.is_integer() else result


def safe_last_rate(case, index):
    try:
        rate = case["history"][-1]["rates"][index]
        return clamp_rate(rate if rate is not None else RATE_FLOOR)
    except (KeyError, IndexError, TypeError, ValueError):
        return RATE_FLOOR


def persistence_rows(case):
    if not isinstance(case, dict):
        case = {}
    trophies = case.get("trophies")
    if not isinstance(trophies, list) or not trophies:
        trophies = [{"id": i} for i in range(1, 56)]
    try:
        horizon_value = case.get("horizon_minutes", 0) if isinstance(case, dict) else 0
        horizon = max(0.0, float(horizon_value)) / 1440.0
    except (TypeError, ValueError):
        horizon = 0.0
    spread = BASELINE_SIGMA * math.sqrt(horizon)
    rows = []
    for index, trophy in enumerate(trophies):
        rate = safe_last_rate(case, index)
        center = logit_rate(rate)
        values = [rounded_rate(logistic(center + z * spread)) for z in Z]
        identifier = trophy.get("id", index + 1) if isinstance(trophy, dict) else index + 1
        rows.append({"id": identifier, "q": values})
    return {"trophies": rows}


def valid_forecast(forecast, case):
    expected = case.get("trophies", [])
    rows = forecast.get("trophies") if isinstance(forecast, dict) else None
    if not isinstance(rows, list) or len(rows) != len(expected):
        return False
    wanted = [t.get("id") for t in expected]
    actual = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"id", "q"}:
            return False
        q = row["q"]
        if not isinstance(q, list) or len(q) != 5:
            return False
        if any(not isinstance(x, (int, float)) or not math.isfinite(x) or x < 0 or x > 100 for x in q):
            return False
        if any(q[i] > q[i + 1] for i in range(4)):
            return False
        actual.append(row["id"])
    return actual == wanted and len(set(actual)) == len(actual)


def write_case(path, output_path):
    case = {}
    try:
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        if not valid_forecast(forecast, case):
            raise ValueError("forecast failed output validation")
    except Exception:
        forecast = persistence_rows(case)
    payload = json.dumps(forecast, ensure_ascii=False, separators=(",", ":")) + chr(10)
    if len(payload.encode("utf-8")) > 65536:
        forecast = persistence_rows(case)
        payload = json.dumps(forecast, ensure_ascii=False, separators=(",", ":")) + chr(10)
    output_path.write_text(payload, encoding="utf-8")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in"), key=lambda item: item.name):
        write_case(path, OUTPUT_DIR / (path.stem + ".out"))


def weighted_line(xs, ys, weights, target_x):
    """Return a recency-weighted linear prediction, residual scale and slope."""
    if len(xs) < 3:
        return None
    total = sum(weights)
    mean_x = sum(w * x for x, w in zip(xs, weights)) / total
    mean_y = sum(w * y for y, w in zip(ys, weights)) / total
    sxx = sum(w * (x - mean_x) ** 2 for x, w in zip(xs, weights))
    if sxx < 1e-10:
        return None
    slope = sum(w * (x - mean_x) * (y - mean_y)
                for x, y, w in zip(xs, ys, weights)) / sxx
    intercept = mean_y - slope * mean_x
    residual = math.sqrt(sum(w * (y - intercept - slope * x) ** 2
                             for x, y, w in zip(xs, ys, weights)) / total)
    leverage = (target_x - mean_x) ** 2 / sxx
    prediction_scale = residual * math.sqrt(1.0 + 1.0 / len(xs) + leverage)
    return intercept + slope * target_x, prediction_scale, slope


def nonincreasing_projection(values):
    """Equal-weight pool-adjacent-violators projection onto a falling sequence."""
    blocks = []
    for value in values:
        blocks.append([float(value), 1])
        while len(blocks) > 1 and blocks[-2][0] < blocks[-1][0]:
            right = blocks.pop()
            left = blocks.pop()
            count = left[1] + right[1]
            blocks.append([(left[0] * left[1] + right[0] * right[1]) / count, count])
    return [block[0] for block in blocks for _ in range(block[1])]


def page_staleness(history):
    """Hours covered by the final run of identical whole-page reads."""
    if len(history) < 2:
        return 0.0
    try:
        last_signature = (history[-1]["players_tracked"], tuple(history[-1]["rates"]))
        last_time = parse_time(history[-1]["at"])
        start_time = last_time
        for read in reversed(history[:-1]):
            signature = (read["players_tracked"], tuple(read["rates"]))
            if signature != last_signature:
                break
            start_time = parse_time(read["at"])
        return max(0.0, (last_time - start_time) / 3600.0)
    except (KeyError, TypeError, ValueError, IndexError):
        return 0.0


def trophy_series(case, index, anchor, cutoff):
    """Collect moving logit-rate and separate earned/tracked-count observations."""
    points = []
    for read in case["history"]:
        try:
            rate = read["rates"][index]
            tracked = float(read["players_tracked"])
            stamp = parse_time(read["at"])
            if rate is None or tracked <= 0 or stamp < cutoff:
                continue
            rate = clamp_rate(rate)
            x = math.log1p(max(0.0, (stamp - anchor) / 3600.0))
            points.append((stamp, x, logit_rate(rate), math.log(max(0.25, rate * tracked / 100.0)),
                           math.log(tracked), tracked, rate))
        except (IndexError, KeyError, TypeError, ValueError):
            continue
    points.sort(key=lambda row: row[0])
    # Repeated page states carry no new trajectory information. Keep their latest timestamp.
    compact = []
    for point in points:
        if compact and abs(point[2] - compact[-1][2]) < 1e-12 and point[5] == compact[-1][5]:
            compact[-1] = point
        else:
            compact.append(point)
    return compact


def forecast_case(case):
    if case.get("levels") != list(LEVELS) or not case.get("history"):
        raise ValueError("case levels or history are invalid")
    trophies = case["trophies"]
    if not isinstance(trophies, list) or not trophies:
        raise ValueError("case has no trophies")
    history = case["history"]
    at_time = parse_time(case["at"])
    target_time = parse_time(case["target"])
    if target_time < at_time:
        raise ValueError("target precedes checkpoint")
    horizon_days = max(0.0, float(case["horizon_minutes"])) / 1440.0
    target_x = math.log1p(max(0.0, (target_time - parse_time("2026-11-19T00:00:00Z")) / 3600.0))
    cutoff = parse_time(history[-1]["at"]) - 24.0 * 3600.0
    stale_hours = page_staleness(history)
    stale_risk = 1.15 + 0.10 * min(1.5, math.log1p(stale_hours / 2.0)) if stale_hours >= 2.0 else 0.0
    refresh_risk = 0.12 * math.sqrt(horizon_days)
    median_logits = []
    spreads = []
    for index, trophy in enumerate(trophies):
        last_rate = safe_last_rate(case, index)
        last_logit = logit_rate(last_rate)
        series = trophy_series(case, index, parse_time("2026-11-19T00:00:00Z"), cutoff)
        x_values = [row[1] for row in series]
        if len(x_values) > 1:
            low, high = min(x_values), max(x_values)
            weights = [0.45 + 0.55 * (x - low) / (high - low) if high > low else 1.0 for x in x_values]
        else:
            weights = [1.0] * len(x_values)
        rate_fit = weighted_line(x_values, [row[2] for row in series], weights, target_x)
        count_fit = weighted_line(x_values, [row[3] for row in series], weights, target_x)
        tracked_fit = weighted_line(x_values, [row[4] for row in series], weights, target_x)
        if rate_fit is None:
            median = last_logit
            residual_scale = 0.22 if len(series) < 3 else 0.0
        else:
            # The line is the main signal. A count/tracked projection adds denominator growth explicitly.
            median = rate_fit[0]
            residual_scale = rate_fit[1]
            if count_fit is not None and tracked_fit is not None:
                count_rate = min(99.0, max(0.01, 100.0 * math.exp(count_fit[0] - tracked_fit[0])))
                median = 0.65 * median + 0.35 * logit_rate(count_rate)
                residual_scale = math.hypot(residual_scale, 0.35 * count_fit[1])
        # Tracking growth makes dilution less certain even when the observed rate fit is smooth.
        growth_scale = 0.0
        if tracked_fit is not None:
            target_delta = max(0.0, target_x - x_values[-1])
            growth_scale = 0.12 * math.sqrt(max(0.0, tracked_fit[2]) * target_delta)
        median = logit_rate(min(99.0, max(0.01, logistic(median))))
        spread = math.sqrt((BASELINE_SIGMA * math.sqrt(horizon_days)) ** 2
                           + residual_scale ** 2 + growth_scale ** 2
                           + refresh_risk ** 2 + stale_risk ** 2 + 0.08 ** 2)
        median_logits.append(median)
        spreads.append(max(0.08, spread))

    quantile_logits = [[median + z * spread for z in Z]
                       for median, spread in zip(median_logits, spreads)]
    # Platinum cannot be more common than any required trophy; chapters 1..4 and finale fall monotonically.
    by_id = {trophy["id"]: index for index, trophy in enumerate(trophies)}
    for level in range(5):
        values = [row[level] for row in quantile_logits]
        platinum = by_id.get(1)
        if platinum is not None:
            required = [value for i, value in enumerate(values) if i != platinum]
            if required:
                values[platinum] = min(values[platinum], min(required))
        chapter_ids = [by_id[i] for i in range(3, 8) if i in by_id]
        if len(chapter_ids) > 1:
            projected = nonincreasing_projection([values[i] for i in chapter_ids])
            for i, value in zip(chapter_ids, projected):
                values[i] = value
        if platinum is not None:
            required = [value for i, value in enumerate(values) if i != platinum]
            if required:
                values[platinum] = min(values[platinum], min(required))
        for index, value in enumerate(values):
            quantile_logits[index][level] = value

    rows = []
    for trophy, values in zip(trophies, quantile_logits):
        quantiles = [rounded_rate(logistic(value)) for value in values]
        for level in range(1, 5):
            quantiles[level] = max(quantiles[level], quantiles[level - 1])
        rows.append({"id": trophy["id"], "q": quantiles})
    return {"trophies": rows}


if __name__ == "__main__":
    main()
