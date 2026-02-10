#!/usr/bin/env python3
"""
process_survey.py - Pre-aggregate survey data for the climate website.

Usage:
    python3 process_survey.py path/to/survey.csv [--embed index.html]

When --embed is used, the script reads index.html, finds
<script id="survey-data" type="application/json"></script>
and replaces its content with the generated JSON.
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime

import pandas as pd

# Filtering constants
ATTENTION_ANSWER = "7"
DATE_CUTOFF = "2026-01-15"
MIN_COUNTRY_N = 100  # minimum responses to include a country individually

# Age group labels (as they appear in the CSV, in display order)
AGE_GROUPS = [
    "18-24 years old",
    "25-34 years old",
    "35-44 years old",
    "45-54 years old",
    "55-64 years old",
    "65+ years old",
]

# Short age labels for JSON keys
AGE_SHORT = {
    "18-24 years old": "18-24",
    "25-34 years old": "25-34",
    "35-44 years old": "35-44",
    "45-54 years old": "45-54",
    "55-64 years old": "55-64",
    "65+ years old": "65+",
}

# Education level labels (CSV value -> short label)
EDU_GROUPS = [
    ("Graduate or professional degree (MA, MS, MBA, PhD, Law Degree, Medical Degree etc)", "Graduate degree"),
    ("University - Bachelors Degree", "Bachelor's degree"),
    ("Some University but no degree", "Some university"),
    ("Vocational or Similar", "Vocational"),
    ("Secondary", "Secondary"),
]
# These get merged into one group
EDU_MERGE = {
    "Some Secondary": "Less than secondary",
    "Primary": "Less than secondary",
    "Less than Primary": "Less than secondary",
}
EDU_SHORT_ORDER = [e[1] for e in EDU_GROUPS] + ["Less than secondary"]

# Country name shortening
COUNTRY_SHORT = {
    "United States of America": "United States",
    "United Kingdom of Great Britain and Northern Ireland": "United Kingdom",
    "United Arab Emirates": "UAE",
    "Congo, Democratic Republic of the": "DR Congo",
    "Iran (Islamic Republic of)": "Iran",
    "Bolivia (Plurinational State of)": "Bolivia",
    "Venezuela (Bolivarian Republic of)": "Venezuela",
    "Tanzania, United Republic of": "Tanzania",
    "Korea, Republic of": "South Korea",
    "Lao People's Democratic Republic": "Laos",
    "Viet Nam": "Vietnam",
    "Russian Federation": "Russia",
    "Moldova, Republic of": "Moldova",
    "Syrian Arab Republic": "Syria",
    "Micronesia (Federated States of)": "Micronesia",
}

# Q3.1 response options (in order)
Q31_OPTIONS = [
    "Climate change is happening",
    "Climate change is not happening",
    "Not sure",
]

# Q3.2 response options
Q32_OPTIONS = [
    "Caused by human activities",
    "Caused by natural changes in the environment",
    "Caused by a mix of human activities and natural changes in the environment",
    "I don't know",
]

# Q4.1 Likert scale options
Q41_OPTIONS = [
    "Strongly agree",
    "Somewhat agree",
    "Neither agree nor disagree",
    "Somewhat disagree",
    "Strongly disagree",
]

# Q7.7 multi-select options (full text for substring matching)
Q77_FULL_OPTIONS = [
    "Given to poor country governments without restrictions",
    "Given directly to people in poor countries",
    "Put in an insurance fund for disasters (like droughts and hurricanes)",
    "Given to poor country governments and communities for climate resilience investments (like seawalls and air conditioning)",
    "Put in a fund for climate resilience investments that people can apply to, controlled by rich countries",
]

Q77_SHORT = [
    "Unrestricted govt aid",
    "Direct to people",
    "Disaster insurance",
    "Resilience investments",
    "Rich-country controlled fund",
]

# Q7.8 allocation columns
Q78_COLS = ["Q7.8_1", "Q7.8_7", "Q7.8_8", "Q7.8_9", "Q7.8_10"]
Q78_LABELS = [
    "Unrestricted govt aid",
    "Direct to people",
    "Disaster insurance",
    "Resilience investments",
    "Rich-country controlled fund",
]

# Question metadata for the JSON output
QUESTIONS_META = {
    "q71": {
        "text": "Support for climate compensation program",
        "options": ["Yes", "No"],
    },
    "q31": {
        "text": "Do you think climate change is happening?",
        "options": ["Happening", "Not happening", "Not sure"],
    },
    "q32": {
        "text": "What causes climate change?",
        "options": ["Human activities", "Natural changes", "Mix of both", "Don't know"],
    },
    "q41": {
        "text": "Country should take measures to fight climate change",
        "options": [
            "Strongly agree",
            "Somewhat agree",
            "Neither",
            "Somewhat disagree",
            "Strongly disagree",
        ],
    },
    "q72": {"text": "Fund via carbon tax", "options": ["Yes", "No"]},
    "q73": {"text": "Fund via wealth tax on billionaires", "options": ["Yes", "No"]},
    "q74": {"text": "Fund via corporate minimum tax", "options": ["Yes", "No"]},
    "q75": {"text": "Fund via general tax revenues", "options": ["Yes", "No"]},
    "q76": {
        "text": "Require emissions reduction commitments",
        "options": ["Yes", "No"],
    },
    "q77": {
        "text": "How should climate fund be spent?",
        "options": Q77_SHORT,
    },
    "q78": {
        "text": "Fund allocation percentages",
        "options": Q78_LABELS,
    },
}


def load_and_filter(csv_path):
    """Load Qualtrics CSV, skip metadata rows, apply quality filters."""
    df = pd.read_csv(csv_path, low_memory=False)

    # First row is question labels, skip it
    data = df.iloc[1:].copy()

    # Remove ImportId metadata rows
    data = data[~data["Q3.1"].str.contains("ImportId", na=False)]

    # Date filter
    data["StartDate"] = pd.to_datetime(data["StartDate"])
    n_before = len(data)
    data = data[data["StartDate"] >= DATE_CUTOFF]
    print(f"Filtered {n_before - len(data)} test responses before {DATE_CUTOFF}")

    # Attention check filter
    n_before = len(data)
    data = data[data["Attention Check"] == ATTENTION_ANSWER]
    print(f"Filtered {n_before - len(data)} failed attention checks")
    print(f"Valid responses: {len(data)}")

    return data


def get_country_info(data):
    """Determine which countries to include individually and build metadata."""
    counts = data["Country"].value_counts()
    countries = []

    for country_name, n in counts.items():
        if n >= MIN_COUNTRY_N:
            short = COUNTRY_SHORT.get(country_name, country_name)
            is_oecd = data[data["Country"] == country_name]["is_oecd"].eq("True").any()
            is_lmic = data[data["Country"] == country_name]["is_lmic"].eq("True").any()
            group = "oecd" if is_oecd else ("lmic" if is_lmic else "other")
            countries.append(
                {"id": short, "name": short, "full_name": country_name, "group": group, "n": int(n)}
            )

    return countries


def count_yes_no(series):
    """Count Yes/No responses, returning [yes_count, no_count]."""
    yes = int((series == "Yes").sum())
    no = int((series == "No").sum())
    return [yes, no]


def count_options(series, options):
    """Count responses matching each option in order."""
    return [int((series == opt).sum()) for opt in options]


def count_q77(series):
    """Count Q7.7 multi-select using substring matching."""
    counts = []
    for full_opt in Q77_FULL_OPTIONS:
        count = int(series.str.contains(full_opt, na=False, regex=False).sum())
        counts.append(count)
    return counts


def mean_q78(subset):
    """Compute mean allocation percentages for Q7.8 fields."""
    # Only include respondents who answered Q7.7
    has_q77 = subset[subset["Q7.7"].notna() & (subset["Q7.7"] != "")]
    if len(has_q77) == 0:
        return [0.0] * len(Q78_COLS), 0

    means = []
    for col in Q78_COLS:
        values = pd.to_numeric(has_q77[col], errors="coerce").fillna(0)
        means.append(round(float(values.mean()), 1))

    return means, int(len(has_q77))


def aggregate_cell(subset):
    """Aggregate all questions for a subset of data. Returns a cell dict."""
    n = len(subset)
    if n == 0:
        return None

    cell = {"n": n}

    # Q7.1 - compensation support
    cell["q71"] = count_yes_no(subset["Q7.1"])

    # Q3.1 - climate beliefs
    cell["q31"] = count_options(subset["Q3.1"], Q31_OPTIONS)

    # Q3.2 - climate causes
    cell["q32"] = count_options(subset["Q3.2"], Q32_OPTIONS)

    # Q4.1 - climate action agreement
    cell["q41"] = count_options(subset["Q4.1"], Q41_OPTIONS)

    # Q7.2-Q7.5 - funding mechanisms
    cell["q72"] = count_yes_no(subset["Q7.2"])
    cell["q73"] = count_yes_no(subset["Q7.3"])
    cell["q74"] = count_yes_no(subset["Q7.4"])
    cell["q75"] = count_yes_no(subset["Q7.5"])

    # Q7.6 - emissions requirements (all respondents)
    cell["q76"] = count_yes_no(subset["Q7.6"])
    cell["n_q76"] = int(cell["q76"][0] + cell["q76"][1])

    # Q7.7 - spending preferences (multi-select)
    cell["q77"] = count_q77(subset["Q7.7"])
    cell["n_q77"] = int(subset["Q7.7"].notna().sum() - (subset["Q7.7"] == "").sum())

    # Q7.8 - allocation percentages
    cell["q78"], cell["n_q78"] = mean_q78(subset)

    return cell


def build_cells(data, countries):
    """Build all aggregation cells for every (group, age, education) combination."""
    cells = {}

    # Build education short label column
    edu_map = {full: short for full, short in EDU_GROUPS}
    edu_map.update(EDU_MERGE)
    data = data.copy()
    data["edu_short"] = data["Education"].map(edu_map).fillna("")

    def add_cells(key_prefix, subset):
        """Add cells for a given group/country across all age and education combinations."""
        # All ages, all education
        cell = aggregate_cell(subset)
        if cell:
            cells[f"{key_prefix}|All|All"] = cell

        # Each age group (all education)
        for age_full in AGE_GROUPS:
            age_short = AGE_SHORT[age_full]
            age_subset = subset[subset["Age"] == age_full]
            cell = aggregate_cell(age_subset)
            if cell and cell["n"] >= 5:
                cells[f"{key_prefix}|{age_short}|All"] = cell

        # Each education group (all ages)
        for edu_short in EDU_SHORT_ORDER:
            edu_subset = subset[subset["edu_short"] == edu_short]
            cell = aggregate_cell(edu_subset)
            if cell and cell["n"] >= 5:
                cells[f"{key_prefix}|All|{edu_short}"] = cell

        # Each age x education combination
        for age_full in AGE_GROUPS:
            age_short = AGE_SHORT[age_full]
            age_subset = subset[subset["Age"] == age_full]
            for edu_short in EDU_SHORT_ORDER:
                combo_subset = age_subset[age_subset["edu_short"] == edu_short]
                cell = aggregate_cell(combo_subset)
                if cell and cell["n"] >= 5:
                    cells[f"{key_prefix}|{age_short}|{edu_short}"] = cell

    # All respondents
    print("Aggregating: All")
    add_cells("All", data)

    # OECD group
    oecd_data = data[data["is_oecd"] == "True"]
    print(f"Aggregating: OECD (n={len(oecd_data)})")
    add_cells("OECD", oecd_data)

    # LMIC group
    lmic_data = data[data["is_lmic"] == "True"]
    print(f"Aggregating: LMIC (n={len(lmic_data)})")
    add_cells("LMIC", lmic_data)

    # Individual countries
    for country_info in countries:
        full_name = country_info["full_name"]
        short_name = country_info["id"]
        country_data = data[data["Country"] == full_name]
        print(f"Aggregating: {short_name} (n={len(country_data)})")
        add_cells(short_name, country_data)

    return cells


def build_json(data, countries, cells):
    """Construct the final JSON structure."""
    output = {
        "meta": {
            "generated": datetime.now().strftime("%Y-%m-%d"),
            "total_n": int(len(data)),
            "countries": [
                {"id": c["id"], "name": c["name"], "group": c["group"], "n": c["n"]}
                for c in countries
            ],
            "age_groups": [AGE_SHORT[ag] for ag in AGE_GROUPS],
            "edu_groups": EDU_SHORT_ORDER,
            "questions": QUESTIONS_META,
        },
        "cells": cells,
    }
    return output


def embed_in_html(json_str, html_path):
    """Replace content of <script id="survey-data"> tag in HTML file."""
    with open(html_path, "r") as f:
        html = f.read()

    pattern = r'(<script\s+id="survey-data"\s+type="application/json">)(.*?)(</script>)'
    match = re.search(pattern, html, re.DOTALL)

    if not match:
        print("ERROR: Could not find <script id=\"survey-data\"> tag in HTML file.")
        print("Make sure the tag exists: <script id=\"survey-data\" type=\"application/json\"></script>")
        sys.exit(1)

    new_html = html[: match.start(2)] + json_str + html[match.end(2) :]

    with open(html_path, "w") as f:
        f.write(new_html)

    print(f"Embedded {len(json_str):,} bytes of JSON into {html_path}")


def main():
    parser = argparse.ArgumentParser(description="Pre-aggregate survey data for the climate website")
    parser.add_argument("csv_path", help="Path to the survey CSV file")
    parser.add_argument("--embed", metavar="HTML_PATH", help="Embed JSON into the specified HTML file")
    parser.add_argument("--output", metavar="JSON_PATH", help="Write JSON to a file (optional)")
    args = parser.parse_args()

    # Load and filter data
    data = load_and_filter(args.csv_path)

    # Determine countries
    countries = get_country_info(data)
    print(f"\nCountries with >= {MIN_COUNTRY_N} responses: {len(countries)}")
    for c in countries:
        print(f"  {c['id']}: n={c['n']} ({c['group']})")

    # Build aggregation cells
    print()
    cells = build_cells(data, countries)
    print(f"\nTotal cells: {len(cells)}")

    # Build final JSON
    output = build_json(data, countries, cells)
    json_str = json.dumps(output, separators=(",", ":"))
    print(f"JSON size: {len(json_str):,} bytes ({len(json_str) / 1024:.1f} KB)")

    # Output
    if args.output:
        with open(args.output, "w") as f:
            f.write(json_str)
        print(f"Written to {args.output}")

    if args.embed:
        embed_in_html(json_str, args.embed)

    if not args.output and not args.embed:
        print("\nNo --output or --embed specified. Use --output FILE or --embed HTML to save results.")


if __name__ == "__main__":
    main()
