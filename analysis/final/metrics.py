"""
Stage 2: the metrics, from the tidy tables of stage 1 (OUT/data) into
OUT/tables (CSV) and OUT/figures (PNG, PDF, and the plotted data as CSV).
Every number the summary reports goes through Ctx.note, with its sample size
and 95% interval, into tables/summary_numbers.csv.

Parts (each a module with run(ctx)): games, h2h, spots, paper, reasons, side,
cross, dataset, seat.
"""
import json
import os

import numpy as np
import pandas as pd

from analysis.stats import bootstrap, wilson

BOOL = ("choice", "valid", "agree", "first", "first_agree", "reasoned", "replied", "has_reason", "any_claim",
        "any_false", "names_chosen", "names_only_other", "shown", "reviewed", "outcomes")


def to_bool(s):
    return s.map({True: True, False: False, "True": True, "False": False, 1: True, 0: False}).astype("boolean")


class Ctx:
    def __init__(self, out, log, roots=None):
        self.out, self.log, self.roots = out, log, roots or {}
        self.data, self.tables = os.path.join(out, "data"), os.path.join(out, "tables")
        os.makedirs(self.tables, exist_ok=True)
        self.numbers = []
        self._frames = {}

    def frame(self, name):
        if name not in self._frames:
            df = pd.read_csv(os.path.join(self.data, f"{name}.csv"), low_memory=False)
            for c in df.columns:
                if c in BOOL or c.startswith(("theme_", "eff_")) or c.endswith(("_offered", "_taken", "_search")):
                    df[c] = to_bool(df[c])
            self._frames[name] = df
        return self._frames[name]

    @property
    def datasets(self):
        return self.frame("datasets")

    def display(self, model):
        d = self.datasets.drop_duplicates("model").set_index("model")["display"]
        return d.get(model, model)

    def write(self, name, df):
        df = pd.DataFrame(df)
        df.to_csv(os.path.join(self.tables, f"{name}.csv"), index=False)
        return df

    def note(self, section, metric, value, n=None, lo=None, hi=None, unit="", detail=""):
        """Record one reported number (with its sample size and 95% interval)."""
        f = lambda x: None if x is None or (isinstance(x, float) and np.isnan(x)) else float(x)
        n = None if n is None or (isinstance(n, float) and np.isnan(n)) else int(n)
        self.numbers.append({"section": section, "metric": metric, "value": f(value), "n": n,
                             "lo": f(lo), "hi": f(hi), "unit": unit, "detail": detail})


def rate(k, n):
    """(rate, lo, hi) with a Wilson interval."""
    lo, hi = wilson(int(k), int(n))
    return (k / n if n else np.nan), lo, hi


def cluster_mean(values, clusters, b=4000):
    """Mean of values with a bootstrap interval over clusters: (est, lo, hi, p vs 0)."""
    df = pd.DataFrame({"v": np.asarray(values, float), "c": np.asarray(clusters)})
    s = df.groupby("c")["v"].agg(["sum", "count"]).to_numpy()
    est, lo, hi, p = bootstrap(s, lambda t: t[..., 0] / t[..., 1], b=b)
    return est, lo, hi, p


def ratio(num, den, clusters, b=4000):
    """sum(num) / sum(den) with a bootstrap interval over clusters."""
    df = pd.DataFrame({"a": np.asarray(num, float), "b": np.asarray(den, float), "c": np.asarray(clusters)})
    s = df.groupby("c")[["a", "b"]].sum().to_numpy()
    est, lo, hi, _ = bootstrap(s, lambda t: t[..., 0] / t[..., 1], b=b)
    return est, lo, hi


def run(out, log, roots=None):
    from analysis.final import (m_cross, m_dataset, m_games, m_h2h, m_paper, m_reasons, m_seat,
                                m_side, m_spots)
    ctx = Ctx(out, log, roots)
    for part in (m_games, m_h2h, m_spots, m_paper, m_reasons, m_side, m_cross, m_dataset, m_seat):
        name = part.__name__.rsplit(".", 1)[-1]
        log(f"metrics: {name}")
        part.run(ctx)
        ctx.write("summary_numbers", ctx.numbers)
        with open(os.path.join(out, "PROGRESS.md"), "a", encoding="utf-8") as f:
            f.write(f"- stage 2 part {name} done\n")
    with open(os.path.join(out, "tables", "summary_numbers.json"), "w", encoding="utf-8") as f:
        json.dump(ctx.numbers, f, indent=1)
    return ctx
