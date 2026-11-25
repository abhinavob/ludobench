"""
Stage 1a: which runs the final analysis uses, under which tidy experiment and
model names, and the complete spot runs.

Every run under the results roots gets a role: "use" (a dataset of its own),
"part" (merged into a dataset: a -final rerun of the spots that changed, or a
part of a split run), "replay" (a game replayed in place of an excluded one),
"seat-check" (the seat-effect check), or "excluded" with the reason.

A dataset is one model (and setting) in one tidy experiment. A spot dataset
asked on spots_40_repaired/ is completed for spots_40_final/: the records of
the 44 spots that differ between the two (spots_40_final/changes.json,
same_as_repaired false) are replaced by the -final rerun's, in the original's
order and columns. A run asked on spots_40_final/ needs no rerun.
"""
import json
import os
import re
from dataclasses import dataclass, field

from analysis.common import REPO, find_runs, lines

# Run folders left out, by experiment/name (whatever the root), with the reason.
EXCLUDED = {
    "main/gpt-oss-20b_heuristic": "partial run (15 of 100 games, cluster)",
    "spots/nvidia-moonshotai-kimi-k3.1-pro": "partial run (53 of 1,040 prompts, NVIDIA)",
    "spots-think/qwen3.5-9b": "thinking qwen3.5:9b timed out (1 of 130 spots)",
    "spots/openrouter-anthropic-claude-opus-5.5": "partial run (35 of 520 spots), superseded by spots-opus130",
}
# Experiment folder prefixes left out entirely.
EXCLUDED_PREFIXES = {"pilot": "pilot run", "sandbox": "sandbox run", "check": "check run",
                     "archive-": "archive run (earlier rules or Ollama version)"}
SEAT_CHECK = ("seat-check", "seat-check-884")
# Tidy experiment of each experiment folder (the 4-player round robins keep their seatings).
EXPERIMENTS = {"main": "main", "hidden": "hidden", "main-outcomes": "main-outcomes",
               "main-history": "main-history", "history": "main-history", "bots": "bots",
               "h2h-2p": "h2h-2p", "h2h-4p/all": "h2h-4p", "h2h-4p-heuristic/all": "h2h-4p-heuristic",
               "spots": "spots", "spots-outcomes": "spots-outcomes", "spots-shuffle": "spots-shuffle",
               "spots-think": "spots-think", "spots-repeat": "spots-repeat",
               "spots-opus130": "spots", "spots-grok130": "spots",
               "spots-argon130": "spots", "spots-argon390": "spots", "spots-argon390b": "spots"}
RERUN = re.compile(r"^(spots(?:-outcomes|-shuffle|-think)?)-final$")
SUBSET_130 = ("spots-opus130", "spots-grok130")     # and every run of --per-category 10

# model id (spec without llm:, provider or host) -> display name, family, maker, size (B, from
# the name; MoE: total), active (B, from the name)
MODELS = {
    "qwen3.5:0.8b": ("Qwen3.5 0.8B", "Qwen3.5", "Alibaba", 0.8, None),
    "qwen3.5:2b": ("Qwen3.5 2B", "Qwen3.5", "Alibaba", 2, None),
    "qwen3.5:4b": ("Qwen3.5 4B", "Qwen3.5", "Alibaba", 4, None),
    "qwen3.5:9b": ("Qwen3.5 9B", "Qwen3.5", "Alibaba", 9, None),
    "qwen3.5:27b": ("Qwen3.5 27B", "Qwen3.5", "Alibaba", 27, None),
    "qwen3.5:122b": ("Qwen3.5 122B", "Qwen3.5", "Alibaba", 122, None),
    "qwen2.5:7b": ("Qwen2.5 7B", "Qwen2.5", "Alibaba", 7, None),
    "qwen.qwen3-235b-a22b-2507-v1:0": ("Qwen3 235B A22B", "Qwen3", "Alibaba", 235, 22),
    "qwen.qwen3-next-80b-a3b": ("Qwen3 Next 80B A3B", "Qwen3", "Alibaba", 80, 3),
    "gemma4:e2b": ("Gemma 4 E2B", "Gemma 4", "Google", 2, None),
    "gemma4:e4b": ("Gemma 4 E4B", "Gemma 4", "Google", 4, None),
    "gemma4:12b": ("Gemma 4 12B", "Gemma 4", "Google", 12, None),
    "gemma4:31b": ("Gemma 4 31B", "Gemma 4", "Google", 31, None),
    "google.gemma-3-4b-it": ("Gemma 3 4B", "Gemma 3", "Google", 4, None),
    "google.gemma-3-12b-it": ("Gemma 3 12B", "Gemma 3", "Google", 12, None),
    "google/gemini-3.5-flash-lite": ("Gemini 3.5 Flash-Lite", "Gemini", "Google", None, None),
    "google/gemini-4-argon": ("Gemini 4 Argon", "Gemini", "Google", None, None),
    "gpt-oss:20b": ("gpt-oss-20b", "gpt-oss", "OpenAI", 20, None),
    "openai.gpt-oss-20b-1:0": ("gpt-oss-20b", "gpt-oss", "OpenAI", 20, None),
    "gpt-oss:120b": ("gpt-oss-120b", "gpt-oss", "OpenAI", 120, None),
    "openai.gpt-oss-120b-1:0": ("gpt-oss-120b", "gpt-oss", "OpenAI", 120, None),
    "openai/gpt-6-luna": ("GPT-6 Luna", "GPT-6", "OpenAI", None, None),
    "llama3.1:8b": ("Llama 3.1 8B", "Llama 3.1", "Meta", 8, None),
    "meta-llama/llama-4-scout": ("Llama 4 Scout", "Llama 4", "Meta", None, None),
    "meta-llama/llama-4-maverick": ("Llama 4 Maverick", "Llama 4", "Meta", None, None),
    "nvidia/nemotron-3.5-lightning-30b-a3b": ("Nemotron 3.5 Lightning 30B A3B", "Nemotron 3.5", "NVIDIA", 30, 3),
    "nvidia/nemotron-3.5-super-vl-120b-a12b": ("Nemotron 3.5 Super 120B A12B", "Nemotron 3.5", "NVIDIA",
                                               120, 12),
    "minimaxai/minimax-m3.1-flash": ("MiniMax M3.1 Flash", "MiniMax", "MiniMax", None, None),
    "global.moonshotai.kimi-k3.1": ("Kimi K3.1", "Kimi", "Moonshot AI", None, None),
    "deepseek.v3.2": ("DeepSeek V3.2", "DeepSeek", "DeepSeek", None, None),
    "deepseek/deepseek-v4.1-flash-1017": ("DeepSeek V4.1 Flash", "DeepSeek", "DeepSeek", None, None),
    "zai.glm-5": ("GLM 5", "GLM", "Z.ai", None, None),
    "z-ai/glm-5.3-flash": ("GLM 5.3 Flash", "GLM", "Z.ai", None, None),
    "mistral.mistral-large-3-675b-instruct": ("Mistral Large 3", "Mistral", "Mistral AI", 675, None),
    "amazon.nova-pro-v1:0": ("Nova Pro", "Nova", "Amazon", None, None),
    "global.amazon.nova-2-lite-v1:0": ("Nova 2 Lite", "Nova", "Amazon", None, None),
    "granite4.2:8b": ("Granite 4.2 8B", "Granite", "IBM", 8, None),
    "olmo-3:7b-instruct": ("OLMo 3 7B", "OLMo", "Ai2", 7, None),
    "anthropic/claude-haiku-5.5": ("Claude Haiku 5.5", "Claude", "Anthropic", None, None),
    "anthropic/claude-sonnet-5.5": ("Claude Sonnet 5.5", "Claude", "Anthropic", None, None),
    "anthropic/claude-opus-5.5": ("Claude Opus 5.5", "Claude", "Anthropic", None, None),
    "x-ai/grok-4.7": ("Grok 4.7", "Grok", "xAI", None, None),
}
BOTS = {"heuristic": "Heuristic bot", "random": "Random bot", "first": "First-listed bot",
        "gt": "Search player (depth 2)"}
# Bedrock on-demand list prices, US$ per million input / output tokens, Asia Pacific (Sydney),
# read from the Bedrock pricing page in November 2026. None: no Sydney price found.
BEDROCK_PRICES = {
    "openai.gpt-oss-120b-1:0": (0.1545, 0.6180), "openai.gpt-oss-20b-1:0": (0.0721, 0.3090),
    "qwen.qwen3-235b-a22b-2507-v1:0": (0.2266, 0.9064), "qwen.qwen3-next-80b-a3b": (0.1545, 1.2360),
    "mistral.mistral-large-3-675b-instruct": (0.5150, 1.5450), "google.gemma-3-12b-it": (0.0927, 0.2987),
    "google.gemma-3-4b-it": (0.0412, 0.0824), "deepseek.v3.2": (0.6386, 1.9055), "zai.glm-5": (1.03, 3.30),
    "global.moonshotai.kimi-k3.1": None, "amazon.nova-pro-v1:0": None, "global.amazon.nova-2-lite-v1:0": None,
}
REASONING_SHARE = 0.10      # a run whose replies carry reasoning this often or more reasons during play


def model_id(spec):
    """(provider, model id) of an LLM spec: llm:qwen3.5:4b, llm://host:port/gemma4:12b, llm:bedrock/..."""
    spec = spec.removeprefix("llm:")
    if spec.startswith("//"):
        return "ollama", spec.split("/", 3)[3]
    provider, _, rest = spec.partition("/")
    if provider in ("bedrock", "openrouter", "nvidia"):
        return provider, rest
    return "ollama", spec


def model_info(spec):
    """{model, display, family, maker, size_b, active_b, provider} of a player spec (LLM or bot)."""
    if spec in BOTS:
        return {"model": spec, "display": BOTS[spec], "family": "bot", "maker": "", "size_b": None,
                "active_b": None, "provider": "bot"}
    provider, mid = model_id(spec)
    display, family, maker, size, active = MODELS[mid]
    return {"model": f"{provider}:{mid}", "display": display, "family": family, "maker": maker,
            "size_b": size, "active_b": active, "provider": provider}


def changed_spots(folder=os.path.join(REPO, "spots_40_final")):
    """The ids of the spots of spots_40_final/ that differ from spots_40_repaired/."""
    with open(os.path.join(folder, "changes.json"), encoding="utf-8") as f:
        return {e["id"] for e in json.load(f)["spots"] if not e["same_as_repaired"]}


def spot_order(folder=os.path.join(REPO, "spots_40_final")):
    """spot id -> its position in the spot files (file name order, then order in the file)."""
    import glob
    ids = [s["id"] for p in sorted(glob.glob(os.path.join(folder, "spots_*.json")))
           for s in json.load(open(p, encoding="utf-8"))]
    return {sid: i for i, sid in enumerate(ids)}


def role(run):
    """(role, tidy experiment or reason) of a run, by its experiment folder and name."""
    where = f"{run.experiment}/{run.name}"
    if where in EXCLUDED:
        return "excluded", EXCLUDED[where]
    for prefix, why in EXCLUDED_PREFIXES.items():
        if run.experiment.startswith(prefix):
            return "excluded", why
    if run.experiment in SEAT_CHECK:
        return "seat-check", "seat-check"
    if re.fullmatch(r".+-replay-s\d+", run.experiment):
        return "replay", run.experiment.rsplit("-replay-", 1)[0]
    m = RERUN.match(run.experiment)
    if m:
        return "part", EXPERIMENTS[m.group(1)]
    if run.experiment.startswith("spots-argon"):
        return "part", "spots"
    if run.experiment in EXPERIMENTS:
        return "use", EXPERIMENTS[run.experiment]
    return "excluded", f"unknown experiment {run.experiment}"


@dataclass
class Dataset:
    experiment: str                 # tidy experiment
    model: str                      # provider:model id, or a bot, or "h2h" for a round robin
    setting: str                    # "" or the reasoning setting (think, effort-low, ...)
    kind: str                       # games or spots
    runs: list = field(default_factory=list)       # source runs, the base first
    subset: str = "all"             # spots asked: all 520, or the 130 of --per-category 10
    info: dict = field(default_factory=dict)

    @property
    def key(self):
        return f"{self.experiment}/{self.model}" + (f"/{self.setting}" if self.setting else "")


def setting_of(run):
    s, p = run.settings, run.meta["players"][str(run.pid)]
    if s.get("think"):
        return "think"
    effort = p.get("effort") or s.get("effort")
    return f"effort-{effort}" if effort and run.experiment.startswith("spots-think") else ""


def llm_spec(run):
    p = run.meta["players"]
    llm = [v for v in p.values() if v.get("provider")]
    return (llm or [p[str(run.pid)]])[0]["model_spec"]


def build(roots):
    """(datasets, run rows): every run with its role, and the datasets the analysis uses."""
    runs = find_runs(roots)
    datasets, rows = {}, []
    for run in runs:
        what, exp = role(run)
        rows.append({"run": run.key, "experiment_folder": run.experiment, "name": run.name, "kind": run.kind,
                     "role": what, "experiment": exp if what in ("use", "part", "replay") else "",
                     "reason": exp if what == "excluded" else "", "commit": (run.meta.get("git") or {}).get("commit")})
        if what not in ("use", "part"):
            continue
        if exp in ("h2h-2p", "h2h-4p", "h2h-4p-heuristic"):
            model = "h2h" if exp != "h2h-2p" else run.name
        else:
            model = model_info(llm_spec(run))["model"]
        d = datasets.setdefault((exp, model, setting_of(run)),
                                Dataset(exp, model, setting_of(run), run.kind))
        base = what == "use" and not run.experiment.startswith("spots-argon")
        d.runs.insert(0, run) if base else d.runs.append(run)
        if run.experiment in SUBSET_130 or (run.settings.get("per_category") == 10
                                            and not run.experiment.startswith("spots-argon")):
            d.subset = "130"
    for d in datasets.values():
        if d.experiment.startswith("h2h"):
            continue
        d.info = model_info(llm_spec(d.runs[0]))
    return sorted(datasets.values(), key=lambda d: d.key), rows, runs


def complete_spots(d, changed=None, order=None):
    """
    The records of a spot dataset on spots_40_final/: its base run with the
    changed spots' records taken from its -final rerun (same order, the base's
    fields), or the parts of a split run in spot order. Returns (records, check).
    """
    changed = changed_spots() if changed is None else changed
    base, *parts = d.runs
    recs = [r for r in lines(os.path.join(base.path, "decisions.jsonl"))]
    check = {"base": base.key, "base_folder": base.settings["spots_folder"], "parts": [p.key for p in parts]}
    if base.experiment.startswith("spots-argon") or d.model.endswith("gemini-4-argon"):
        order = spot_order() if order is None else order
        recs += [r for p in parts for r in lines(os.path.join(p.path, "decisions.jsonl"))]
        recs.sort(key=lambda r: (order[r["spot_id"]], r["config"]["legal_moves_shown"]))
        check["replaced"] = 0
        return recs, check
    if base.settings["spots_folder"] != "spots_40_repaired":
        check["replaced"] = 0
        if parts:
            raise ValueError(f"{d.key}: a rerun of a run already on {base.settings['spots_folder']}")
        return recs, check
    asked = {r["spot_id"] for r in recs}
    expected = {(s, c) for s, c in ((r["spot_id"], r["config"]["legal_moves_shown"]) for r in recs) if s in changed}
    if not expected and not parts:              # none of its spots changed (e.g. a repeat run)
        check["replaced"] = 0
        return recs, check
    if len(parts) != 1:
        raise ValueError(f"{d.key}: expected one -final rerun, found {len(parts)}")
    rerun = {(r["spot_id"], r["config"]["legal_moves_shown"]): r
             for r in lines(os.path.join(parts[0].path, "decisions.jsonl"))}
    if set(rerun) != expected:
        raise ValueError(f"{d.key}: the rerun holds {len(rerun)} prompts, expected the {len(expected)} "
                         f"of the changed spots asked")
    keys = list(recs[0])
    recs = [{k: rerun[(r["spot_id"], r["config"]["legal_moves_shown"])].get(k) for k in keys}
            if r["spot_id"] in changed else r for r in recs]
    check.update(replaced=len(rerun), changed_asked=len(asked & changed))
    return recs, check
