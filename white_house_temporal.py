"""
white_house_temporal.py
-----------------------
Analyzes White House article metrics over time to detect shifts in writing
style, tone, or authorship. Looks for abrupt changes in readability,
sentiment, rhetoric intensity, and stylometric features that might indicate
staff changes or rotating writers.

Reads from the silver data lake CSVs produced by:
  - white_house_analysis.py  (sentiment, rhetoric, political scores)
  - white_house_stylometry.py (stylometric features)

Outputs a combined temporal analysis CSV and a text-based summary report.

Dependencies: Standard library only (csv, json, os, math, statistics)
"""

import csv
import os
import math
from datetime import datetime
from statistics import mean, stdev, median


# ─── Date Parsing ────────────────────────────────────────────────────────────

MONTH_MAP = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12
}


def parse_date(date_str):
    """
    Parse date strings like 'January 20, 2026' or 'February 2, 2025'.
    Returns a datetime object or None if parsing fails.
    """
    if not date_str or date_str == "N/A":
        return None
    try:
        parts = date_str.strip().replace(",", "").split()
        if len(parts) == 3:
            month = MONTH_MAP.get(parts[0].lower())
            day = int(parts[1])
            year = int(parts[2])
            if month:
                return datetime(year, month, day)
    except (ValueError, KeyError):
        pass
    return None


def format_month(dt):
    """Format datetime as 'YYYY-MM' string."""
    return f"{dt.year}-{dt.month:02d}"


# ─── CSV Loading ─────────────────────────────────────────────────────────────

def load_csv(filepath):
    """Load a CSV file into a list of dicts. Returns empty list if missing."""
    if not os.path.exists(filepath):
        print(f"  File not found: {filepath}")
        return []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def safe_float(value):
    """Convert a string to float, returning None if not possible."""
    if value is None or value == "" or value == "None":
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


# ─── Change Point Detection ─────────────────────────────────────────────────

def detect_shifts(timeseries, threshold_sigma=1.5):
    """
    Simple change-point detection using a rolling z-score approach.
    
    For each data point, computes how many standard deviations it is from
    the rolling mean of the previous window. Points exceeding the threshold
    are flagged as potential shift points.
    
    Args:
        timeseries: list of (date_key, value) tuples, sorted by date
        threshold_sigma: number of std devs to consider a shift
    
    Returns:
        list of (date_key, value, z_score) for flagged points
    """
    if len(timeseries) < 4:
        return []

    window_size = max(3, len(timeseries) // 4)
    shifts = []

    for i in range(window_size, len(timeseries)):
        window_values = [v for _, v in timeseries[i - window_size:i]]
        current_key, current_val = timeseries[i]

        if len(window_values) < 2:
            continue

        w_mean = mean(window_values)
        w_std = stdev(window_values)

        if w_std == 0:
            continue

        z_score = (current_val - w_mean) / w_std

        if abs(z_score) >= threshold_sigma:
            shifts.append((current_key, current_val, round(z_score, 2)))

    return shifts


# ─── Temporal Aggregation ────────────────────────────────────────────────────

def aggregate_by_month(rows, date_field, metric_fields):
    """
    Group rows by month and compute mean values for each metric field.
    
    Returns:
        dict of { metric_name: [(month_key, avg_value), ...] } sorted by month
    """
    monthly_buckets = {}

    for row in rows:
        dt = parse_date(row.get(date_field, ""))
        if dt is None:
            continue

        month_key = format_month(dt)

        if month_key not in monthly_buckets:
            monthly_buckets[month_key] = {field: [] for field in metric_fields}

        for field in metric_fields:
            val = safe_float(row.get(field))
            if val is not None:
                monthly_buckets[month_key][field].append(val)

    # Compute averages and build sorted timeseries per metric
    result = {}
    sorted_months = sorted(monthly_buckets.keys())

    for field in metric_fields:
        series = []
        for month in sorted_months:
            values = monthly_buckets[month][field]
            if values:
                series.append((month, round(mean(values), 4)))
        result[field] = series

    return result, sorted_months, monthly_buckets


# ─── Report Generation ──────────────────────────────────────────────────────

def generate_report(analysis_series, stylometry_series, analysis_shifts, stylometry_shifts, output_path):
    """Generate a human-readable text report of the temporal analysis."""
    lines = []
    lines.append("=" * 70)
    lines.append("WHITE HOUSE ARTICLE TEMPORAL ANALYSIS REPORT")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 70)

    # ── Analysis Metrics Summary ──
    if analysis_series:
        lines.append("\n── ANALYSIS METRICS (Monthly Averages) ──\n")
        for metric, series in analysis_series.items():
            if not series:
                continue
            values = [v for _, v in series]
            lines.append(f"  {metric}:")
            lines.append(f"    Range:  {min(values):.4f} — {max(values):.4f}")
            lines.append(f"    Mean:   {mean(values):.4f}")
            lines.append(f"    Median: {median(values):.4f}")
            if len(values) > 1:
                lines.append(f"    StdDev: {stdev(values):.4f}")
            lines.append("")

    # ── Stylometry Metrics Summary ──
    if stylometry_series:
        lines.append("\n── STYLOMETRY METRICS (Monthly Averages) ──\n")
        for metric, series in stylometry_series.items():
            if not series:
                continue
            values = [v for _, v in series]
            lines.append(f"  {metric}:")
            lines.append(f"    Range:  {min(values):.4f} — {max(values):.4f}")
            lines.append(f"    Mean:   {mean(values):.4f}")
            lines.append(f"    Median: {median(values):.4f}")
            if len(values) > 1:
                lines.append(f"    StdDev: {stdev(values):.4f}")
            lines.append("")

    # ── Detected Shifts ──
    all_shifts = []
    for metric, shifts in {**analysis_shifts, **stylometry_shifts}.items():
        for month, value, z_score in shifts:
            all_shifts.append((month, metric, value, z_score))

    all_shifts.sort(key=lambda x: (x[0], abs(x[3])), reverse=True)

    lines.append("\n── DETECTED STYLE SHIFTS (|z-score| >= 1.5) ──\n")
    if all_shifts:
        lines.append(f"  {'Month':<10} {'Metric':<30} {'Value':>10} {'Z-Score':>10}")
        lines.append(f"  {'-'*10} {'-'*30} {'-'*10} {'-'*10}")
        for month, metric, value, z in all_shifts:
            direction = "▲" if z > 0 else "▼"
            lines.append(f"  {month:<10} {metric:<30} {value:>10.4f} {direction}{abs(z):>9.2f}")
    else:
        lines.append("  No significant shifts detected.")

    # ── Interpretation Guidance ──
    lines.append("\n── INTERPRETATION GUIDE ──\n")
    lines.append("  • A sudden shift in avg_sentence_length or avg_word_length may")
    lines.append("    indicate a different author or editor took over.")
    lines.append("  • Changes in function word ratios (fw_the, fw_of, etc.) are the")
    lines.append("    strongest signals of authorship change, as these are unconscious.")
    lines.append("  • Shifts in caps_rate or exclamation_rate may reflect editorial")
    lines.append("    style changes rather than authorship changes.")
    lines.append("  • Multiple metrics shifting simultaneously in the same month is")
    lines.append("    a strong indicator of a writing staff change.")
    lines.append("")

    report_text = "\n".join(lines)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(report_text)
    return report_text


# ─── Combined Monthly CSV ───────────────────────────────────────────────────

def save_monthly_csv(analysis_series, stylometry_series, sorted_months_analysis,
                     sorted_months_stylometry, monthly_counts_analysis, output_path):
    """Save a combined monthly aggregation CSV for downstream analysis."""
    all_months = sorted(set(
        (sorted_months_analysis or []) + (sorted_months_stylometry or [])
    ))

    if not all_months:
        print("  No monthly data to save.")
        return

    # Collect all metric names
    all_metrics = []
    if analysis_series:
        all_metrics += list(analysis_series.keys())
    if stylometry_series:
        all_metrics += list(stylometry_series.keys())

    # Build lookup dicts:  metric -> { month: value }
    lookup = {}
    for metric, series in {**(analysis_series or {}), **(stylometry_series or {})}.items():
        lookup[metric] = {month: val for month, val in series}

    # Count articles per month from analysis data
    article_counts = {}
    if monthly_counts_analysis:
        for month, bucket in monthly_counts_analysis.items():
            # Use the first available metric to count how many entries
            for field, values in bucket.items():
                if values:
                    article_counts[month] = len(values)
                    break

    fieldnames = ["month", "article_count"] + all_metrics
    rows = []
    for month in all_months:
        row = {"month": month, "article_count": article_counts.get(month, "")}
        for metric in all_metrics:
            row[metric] = lookup.get(metric, {}).get(month, "")
        rows.append(row)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Monthly aggregation saved to {output_path}")


# ─── Main Pipeline ───────────────────────────────────────────────────────────

def run_temporal_analysis(analysis_csv_path, stylometry_csv_path, output_dir):
    """
    Main entry point. Loads silver-layer CSVs, aggregates by month,
    detects shifts, and produces a report + combined CSV.
    """
    print("Loading analysis results...")
    analysis_rows = load_csv(analysis_csv_path)
    print(f"  Loaded {len(analysis_rows)} rows from analysis CSV.")

    print("Loading stylometry features...")
    stylometry_rows = load_csv(stylometry_csv_path)
    print(f"  Loaded {len(stylometry_rows)} rows from stylometry CSV.")

    # ── Define which metrics to track over time ──
    analysis_metrics = [
        "sentiment_compound", "positive_sentiment", "negative_sentiment",
        "flesch_kincaid_grade", "top_rhetoric_score", "top_political_score",
    ]
    # Dynamically include all rhetoric_ and political_ columns
    if analysis_rows:
        for key in analysis_rows[0].keys():
            if (key.startswith("rhetoric_") or key.startswith("political_")) and key not in analysis_metrics:
                analysis_metrics.append(key)

    stylometry_metrics = [
        "avg_sentence_length", "std_sentence_length",
        "avg_word_length", "std_word_length",
        "type_token_ratio", "hapax_ratio", "yules_k",
        "comma_rate", "semicolon_rate", "colon_rate",
        "exclamation_rate", "question_rate", "dash_rate",
        "passive_voice_ratio", "flesch_kincaid_grade", "flesch_reading_ease",
        "avg_paragraph_length", "caps_rate",
    ]
    # Include function word columns dynamically
    if stylometry_rows:
        for key in stylometry_rows[0].keys():
            if key.startswith("fw_") and key not in stylometry_metrics:
                stylometry_metrics.append(key)

    # ── Aggregate by month ──
    print("\nAggregating analysis metrics by month...")
    analysis_series, sorted_months_a, monthly_buckets_a = aggregate_by_month(
        analysis_rows, "date", analysis_metrics
    ) if analysis_rows else ({}, [], {})

    print("Aggregating stylometry metrics by month...")
    stylometry_series, sorted_months_s, monthly_buckets_s = aggregate_by_month(
        stylometry_rows, "date", stylometry_metrics
    ) if stylometry_rows else ({}, [], {})

    # ── Detect shifts ──
    print("\nRunning change-point detection...")
    analysis_shifts = {}
    for metric, series in analysis_series.items():
        shifts = detect_shifts(series, threshold_sigma=1.5)
        if shifts:
            analysis_shifts[metric] = shifts
            print(f"  {metric}: {len(shifts)} shift(s) detected")

    stylometry_shifts = {}
    for metric, series in stylometry_series.items():
        shifts = detect_shifts(series, threshold_sigma=1.5)
        if shifts:
            stylometry_shifts[metric] = shifts
            print(f"  {metric}: {len(shifts)} shift(s) detected")

    total_shifts = sum(len(s) for s in analysis_shifts.values()) + \
                   sum(len(s) for s in stylometry_shifts.values())
    print(f"\n  Total shifts detected: {total_shifts}")

    # ── Save outputs ──
    os.makedirs(output_dir, exist_ok=True)

    report_path = os.path.join(output_dir, "temporal_analysis_report.txt")
    print(f"\nGenerating report...\n")
    generate_report(analysis_series, stylometry_series,
                    analysis_shifts, stylometry_shifts, report_path)

    monthly_csv_path = os.path.join(output_dir, "monthly_aggregated_metrics.csv")
    save_monthly_csv(analysis_series, stylometry_series,
                     sorted_months_a, sorted_months_s,
                     monthly_buckets_a, monthly_csv_path)


if __name__ == "__main__":
    base = os.getcwd()
    silver_dir = os.path.join(base, ".data_lake", "02_Silver", "white_house")

    analysis_csv = os.path.join(silver_dir, "article_analysis_results.csv")
    stylometry_csv = os.path.join(silver_dir, "stylometry_features.csv")
    output_dir = os.path.join(silver_dir, "temporal_analysis")

    run_temporal_analysis(analysis_csv, stylometry_csv, output_dir)
