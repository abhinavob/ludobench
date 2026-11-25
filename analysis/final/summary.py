"""
RESULTS_SUMMARY.md: the written part (OUT/summary_head.md, if present), then
every number the metrics recorded (tables/summary_numbers.csv) with its 95%
interval and sample size, the exclusions (data/runs.csv) and the datasets.

    python -m analysis.final.summary --out ../final_results
"""
import argparse
import os

import pandas as pd


def fmt(v, unit=""):
    if pd.isna(v):
        return ""
    if unit in ("count",) or float(v).is_integer() and abs(v) >= 2:
        return f"{int(v):,}"
    if "share" in unit or unit in ("difference", "mean over models"):
        return f"{v * 100:.1f}%" if unit != "difference" else f"{v * 100:+.1f} pts"
    return f"{v:.3g}"


def numbers_section(out):
    t = pd.read_csv(os.path.join(out, "tables", "summary_numbers.csv"))
    lines = ["## Every reported number", "",
             "Value, 95% interval and sample size (n: games, decisions, spots, prompts or models, as the unit says). "
             "Shares are percentages; differences are percentage points.", ""]
    for section, x in t.groupby("section", sort=False):
        lines += [f"### {section}", "", "| metric | value | 95% interval | n | detail |", "|---|---|---|---|---|"]
        for _, r in x.iterrows():
            ci = f"{fmt(r['lo'], r['unit'])} to {fmt(r['hi'], r['unit'])}" if pd.notna(r["lo"]) else ""
            n = "" if pd.isna(r["n"]) else f"{int(r['n']):,}"
            detail = "" if pd.isna(r["detail"]) else str(r["detail"])
            lines.append(f"| {r['metric']} | {fmt(r['value'], r['unit'] if pd.notna(r['unit']) else '')} | {ci} | "
                         f"{n} | {detail} |")
        lines.append("")
    return lines


def exclusions_section(out):
    runs = pd.read_csv(os.path.join(out, "data", "runs.csv"))
    ex = runs[runs["role"] == "excluded"]
    lines = ["## Runs left out", "", f"{len(ex)} of {len(runs)} run folders are left out:", "",
             "| reason | runs |", "|---|---|"]
    for reason, x in ex.groupby("reason"):
        names = x["run"].tolist()
        lines.append(f"| {reason} | {len(names)}: " + ", ".join(f"`{n}`" for n in names[:60]) +
                     (" ..." if len(names) > 60 else "") + " |")
    other = runs[runs["role"].isin(["seat-check", "replay"])]
    lines += ["", "Used apart from the datasets: " + ", ".join(f"`{r}` ({k})" for r, k in zip(other["run"], other["role"])),
              ""]
    return lines


def datasets_section(out):
    ds = pd.read_csv(os.path.join(out, "data", "datasets.csv"))
    lines = ["## Datasets", "", "| experiment | model | setting | subset | records / games | reasoning group | cost (US$) |",
             "|---|---|---|---|---|---|---|"]
    for _, r in ds.iterrows():
        n = r["records"] if pd.notna(r.get("records")) else r.get("games")
        cost = "" if pd.isna(r.get("cost_usd")) else f"{float(r['cost_usd']):.2f}"
        lines.append(f"| {r['experiment']} | {r['display'] if pd.notna(r['display']) else r['model']} | "
                     f"{'' if pd.isna(r['setting']) else r['setting']} | {r['subset']} | "
                     f"{'' if pd.isna(n) else int(n)} | {'' if pd.isna(r.get('reasoning_group')) else r['reasoning_group']} | "
                     f"{cost} |")
    return lines + [""]


def write(out):
    head = os.path.join(out, "summary_head.md")
    text = open(head, encoding="utf-8").read().rstrip("\n").split("\n") + [""] if os.path.exists(head) else \
        ["# LudoBench: results summary", ""]
    text += numbers_section(out) + exclusions_section(out) + datasets_section(out)
    with open(os.path.join(out, "RESULTS_SUMMARY.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(text) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip())
    ap.add_argument("--out", required=True)
    write(os.path.abspath(ap.parse_args().out))


if __name__ == "__main__":
    main()
