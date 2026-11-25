"""
Stage 1: write the clean data to OUT/data. Each table is written once; a rerun
skips the tables already there (delete one to rebuild it).

  runs.csv            every run found, its role, tidy experiment, and why it is
                      left out
  datasets.csv        one row per dataset: model names, provider, family,
                      size, setting, spot subset, records and games, what was
                      sent for reasoning and how often replies carried it, the
                      reasoning group, and the cost
  spot_reference.csv, spot_decisions.csv, games.csv, game_decisions.csv,
  h2h_players.csv     see tidy.py
  spots/<experiment>/<model>[_<setting>].jsonl
                      the complete spot runs on spots_40_final/ (catalog.py)
  completion.csv      per spot dataset: base run, parts, records replaced
"""
import csv
import glob
import json
import os

from analysis.common import lines
from analysis.final import catalog as C
from analysis.final import tidy as T


def exists(out, name):
    return os.path.exists(os.path.join(out, f"{name}.csv"))


def write(out, name, rows, fields=None):
    os.makedirs(out, exist_ok=True)
    keys = fields or list(dict.fromkeys(k for r in rows for k in r))
    tmp = os.path.join(out, f"{name}.csv.tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, os.path.join(out, f"{name}.csv"))


def read(out, name):
    with open(os.path.join(out, f"{name}.csv"), encoding="utf-8") as f:
        return list(csv.DictReader(f))


def file_name(ds):
    return (ds.model.replace(":", "_").replace("/", "_") + (f"_{ds.setting}" if ds.setting else ""))


def cost(info, rows, tokens=("prompt_tokens", "completion_tokens")):
    """(US$, status) of a dataset from its rows: OpenRouter's reported cost, Bedrock list price, else free."""
    provider, mid = info["provider"], info["model"].split(":", 1)[1]
    if provider == "openrouter":
        return sum(float(r.get("cost_usd") or 0) for r in rows), "reported by OpenRouter"
    if provider == "bedrock":
        price = C.BEDROCK_PRICES.get(mid)
        if price is None:
            return None, "unknown (no Sydney list price)"
        return sum((r.get(tokens[0]) or 0) * price[0] + (r.get(tokens[1]) or 0) * price[1]
                   for r in rows) / 1e6, "estimated (Bedrock list price x tokens)"
    return 0.0, "free (NVIDIA trial API)" if provider == "nvidia" else "free (local)"


def sent(run):
    p = run.meta["players"][str(run.pid)]
    return {"sent_reasoning": json.dumps(p.get("reasoning", {"think": p.get("think")})),
            "sent_effort": p.get("effort"), "thinks": p.get("thinks")}


def build(roots, out, log=print):
    data = os.path.join(out, "data")
    datasets, run_rows, runs = C.build(roots)
    if not exists(data, "runs"):
        write(data, "runs", run_rows)
    spots = [s for p in sorted(glob.glob(os.path.join(C.REPO, "spots_40_final", "spots_*.json")))
             for s in json.load(open(p, encoding="utf-8"))]
    if not exists(data, "spot_reference"):
        log("spot reference: searching every spot at depths 2 and 1")
        write(data, "spot_reference", T.spot_reference(spots))
    ref = {r["spot_id"]: {**r, "n_players": int(r["n_players"]), "n_groups": int(r["n_groups"]),
                          "reviewed": r["reviewed"] == "True",
                          "target": int(r["target"]) if r["target"] else None,
                          "pair_id": r["pair_id"] or None, "history": r["history"] or None}
           for r in read(data, "spot_reference")}
    if not exists(data, "spot_decisions"):
        rows, completion = [], []
        for ds in datasets:
            if ds.kind != "spots":
                continue
            records, srows, check = T.dataset_spots(ds, ref)
            folder = os.path.join(data, "spots", ds.experiment)
            os.makedirs(folder, exist_ok=True)
            with open(os.path.join(folder, file_name(ds) + ".jsonl"), "w", encoding="utf-8") as f:
                f.writelines(json.dumps(r) + "\n" for r in records)
            completion.append({"dataset": ds.key, **check, "records": len(records),
                               "spots": len({r["spot_id"] for r in records})})
            rows += srows
            log(f"spots {ds.key}: {len(records)} records, {check.get('replaced', 0)} replaced")
        write(data, "spot_decisions", rows)
        write(data, "completion", completion)
    if not exists(data, "spot_meta"):
        meta = []
        for ds in datasets:
            if ds.kind == "spots":
                records = list(lines(os.path.join(data, "spots", ds.experiment, file_name(ds) + ".jsonl")))
                n, k, share, group = T.reasoning(records)
                usd, status = cost(ds.info, records)
                meta.append({"dataset": ds.key, "records": len(records), "replies": n, "replies_reasoning": k,
                             "reasoning_share": share, "reasoning_group": group, "cost_usd": usd,
                             "cost_status": status, **sent(ds.runs[0])})
        write(data, "spot_meta", meta)
    if not exists(data, "games"):
        grows, drows, gmeta = [], [], []
        for ds in datasets:
            if ds.kind != "games" or ds.experiment.startswith("h2h"):
                continue
            g, d, notes, (n, k, share, group) = T.dataset_games(ds, runs)
            grows += g
            drows += d
            usd, status = cost(ds.info, g) if ds.info["provider"] != "bot" else (0.0, "bot")
            gmeta.append({"dataset": ds.key, "games": len(g), "decisions": len(d), "replies": n,
                          "replies_reasoning": k, "reasoning_share": share,
                          "reasoning_group": group if ds.info["provider"] != "bot" else "bot",
                          "excluded_games": ";".join(notes["excluded"]), "replaced_by": ";".join(notes["replaced_by"]),
                          "cost_usd": usd, "cost_status": status,
                          **(sent(ds.runs[0]) if ds.info["provider"] != "bot" else {})})
            log(f"games {ds.key}: {len(g)} games, {len(d)} decisions")
        write(data, "game_decisions", drows)
        write(data, "game_meta", gmeta)
        write(data, "games", grows)
    if not exists(data, "h2h_players"):
        rows = [r for ds in datasets if ds.experiment.startswith("h2h") for r in T.h2h_rows(ds, runs)]
        write(data, "h2h_players", rows)
    if not exists(data, "datasets"):
        extra = {r["dataset"]: r for r in read(data, "spot_meta") + read(data, "game_meta")}
        rows = []
        for ds in datasets:
            info = ds.info or {"model": ds.model, "display": ds.model, "provider": "bedrock"}
            rows.append({"dataset": ds.key, "experiment": ds.experiment, "kind": ds.kind, "model": ds.model,
                         "display": info.get("display"), "family": info.get("family"), "maker": info.get("maker"),
                         "provider": info.get("provider"), "size_b": info.get("size_b"),
                         "active_b": info.get("active_b"), "setting": ds.setting, "subset": ds.subset,
                         "runs": ";".join(r.key for r in ds.runs), **extra.get(ds.key, {})})
        write(data, "datasets", rows)
    return datasets, runs
