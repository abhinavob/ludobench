"""
List the Bedrock models and inference profiles our region offers, or send one
test request, with the key in AWS_BEARER_TOKEN_BEDROCK (never printed).

    python analysis/bedrock_models.py                     # model and profile IDs
    python analysis/bedrock_models.py --test au.anthropic.claude-haiku-5-5

A listed ID may still be refused for an account; --test sends one short request
with run_games.py's settings for that model (API, reasoning fields, schema,
ceiling; one attempt) and prints the reply or the API's error.
"""
import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from llm_backends import PROVIDER_DEFAULTS, make_llm  # noqa: E402
import run_games  # noqa: E402

BEDROCK = PROVIDER_DEFAULTS["bedrock"]


def get(path):
    """GET a Bedrock control-plane path with the key; raises SystemExit on an error status."""
    req = urllib.request.Request(BEDROCK["control_url"] + path, headers={
        "Authorization": f"Bearer {os.environ[BEDROCK['key_env']]}"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise SystemExit(f"{path}: HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:300]}")


def listing():
    """Lines naming every foundation model and inference profile of the region."""
    lines = ["foundation models (ID, provider, inference types, lifecycle):"]
    for m in sorted(get("/foundation-models")["modelSummaries"], key=lambda m: m["modelId"]):
        lines.append(f"  {m['modelId']}  {m.get('providerName', '')}  "
                     f"{','.join(m.get('inferenceTypesSupported', []))}  "
                     f"{(m.get('modelLifecycle') or {}).get('status', '')}")
    profiles, token = [], ""
    while True:
        page = get("/inference-profiles?maxResults=1000"
                   + (f"&nextToken={urllib.parse.quote(token)}" if token else ""))
        profiles += page.get("inferenceProfileSummaries", [])
        token = page.get("nextToken")
        if not token:
            break
    lines.append("inference profiles (ID, status):")
    lines += [f"  {p['inferenceProfileId']}  {p.get('status', '')}"
              for p in sorted(profiles, key=lambda p: p["inferenceProfileId"])]
    return lines


def test(model):
    """One request to llm:bedrock/MODEL with run_games.py's settings, described in lines."""
    spec = f"llm:bedrock/{model}"
    player = run_games.player_info([spec], [0])[0]
    kwargs, record = run_games.llm_setup(player, False, False, None, run_games.RELEASED)
    call = make_llm(spec, **dict(kwargs, retries=1))
    lines = [f"{spec}: {record['api']}, reasoning fields {record['reasoning']}, "
             f"schema sent {record['schema_sent']}"]
    try:
        reply = call('Reply with the JSON object {"reason": "test", "piece": 0}.')
    except RuntimeError as e:
        return lines + [f"refused or failed: {e}"]
    return lines + [f"finish {call.last_finish_reason}; tokens in {call.last_prompt_tokens}, "
                    f"out {call.last_completion_tokens}, reasoning {call.last_thinking_tokens}; "
                    f"reasoning text {len(call.last_reasoning)} characters",
                    f"reply: {reply}"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip())
    ap.add_argument("--test", metavar="MODEL", help="send one request to this model or profile ID")
    args = ap.parse_args()
    if not os.environ.get(BEDROCK["key_env"]):
        ap.error(f"set {BEDROCK['key_env']}")
    print("\n".join(test(args.test) if args.test else listing()))


if __name__ == "__main__":
    main()
