"""
How many tokens a model thinks on Ludo spots, to set THINKING_CEILING in
llm_backends.py. Not used by runs.

Sends K spots from spots_40/ (a fixed random sample) to the model with thinking
on, a 32,768-token ceiling, our rules, the legal moves shown and the JSON reply,
and reports the thinking tokens at the 50th/90th/95th/99th percentiles (nearest
rank), seconds per call, and the share of valid replies (JSON naming a legal
piece). The model must be served by Ollama.

Run from the repo root:  python analysis/measure_thinking.py qwen3:8b --spots 20
"""
import argparse
import glob
import json
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm_agent import LLMAgent
from llm_backends import make_llm, model_info, thinking_setting
from spot_evaluation import SpotEnv, compute_legal_moves

CEILING = 32_768


def percentile(values, p):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(p / 100 * len(ordered)) - 1)]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("model", help="Ollama model, e.g. qwen3:8b")
    ap.add_argument("--spots", type=int, default=20, help="number of spots K")
    ap.add_argument("--host", default="http://127.0.0.1:11434")
    ap.add_argument("--seed", type=int, default=0, help="seed of the spot sample")
    args = ap.parse_args()

    spots = [s for f in sorted(glob.glob("spots_40/*.json")) for s in json.load(open(f))]
    spots = random.Random(args.seed).sample(spots, args.spots)
    think, _ = thinking_setting(args.model, model_info(args.host, args.model), think=True)
    host = args.host.split("//", 1)[1]
    call = make_llm(f"ollama_native://{host}/{args.model}", max_tokens=CEILING, stop=None,
                    think=think, structured=True, reason_max_chars=500, timeout=3600)

    rows = []
    for s in spots:
        env = SpotEnv(s["players"], {int(k): v for k, v in s["tokens"].items()})
        legal = compute_legal_moves(env, s["current_player"], s["dice"])
        t0 = time.perf_counter()
        LLMAgent(call).select_move(env, s["current_player"], s["dice"], legal)
        rows.append({"spot": s["id"], "seconds": time.perf_counter() - t0,
                     "thinking_tokens": call.last_thinking_tokens,
                     "output_tokens": call.last_completion_tokens,
                     "finish": call.last_finish_reason, "valid": not env.last_llm_fallback})
        print(f"  {s['id']:<34} thinking {rows[-1]['thinking_tokens']:>6}  "
              f"{rows[-1]['seconds']:6.1f}s  {rows[-1]['finish']}  "
              f"{'valid' if rows[-1]['valid'] else 'INVALID'}")

    thinking = [r["thinking_tokens"] for r in rows]
    print(f"\n{args.model}, {len(rows)} spots, ceiling {CEILING}")
    print("thinking tokens  " + "  ".join(f"p{p} {percentile(thinking, p)}"
                                          for p in (50, 90, 95, 99)))
    print(f"seconds per call {sum(r['seconds'] for r in rows) / len(rows):.1f}")
    print(f"valid replies    {sum(r['valid'] for r in rows)}/{len(rows)}   "
          f"stopped at the ceiling {sum(r['finish'] == 'length' for r in rows)}")


if __name__ == "__main__":
    main()
