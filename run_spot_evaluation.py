import argparse
import json

from spot_evaluation import evaluate_spot


def parse_json(value, name):
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON for {name}: {exc}") from exc


def main():
    parser = argparse.ArgumentParser(
        description="Run a single Ludo spot evaluation (LLM + Heuristic)."
    )
    parser.add_argument("--players", required=True, help='JSON list, e.g. "[0,2]"')
    parser.add_argument(
        "--tokens",
        required=True,
        help='JSON dict, e.g. {"0":[-1,0,14,53],"2":[-1,26,30,64]}',
    )
    parser.add_argument("--dice", required=True, type=int, help="Dice roll (1-6)")
    parser.add_argument("--current-player", required=True, type=int, help="Player id")
    parser.add_argument(
        "--persona",
        default="",
        help="Optional persona text to inject into prompt",
    )

    args = parser.parse_args()

    players = parse_json(args.players, "players")
    tokens_raw = parse_json(args.tokens, "tokens")
    tokens = {int(k): v for k, v in tokens_raw.items()}
    from real_llm import real_llm

    llm_call = real_llm

    result = evaluate_spot(
        players=players,
        tokens=tokens,
        dice=args.dice,
        current_player=args.current_player,
        legal_moves=None,
        llm_call=llm_call,
        persona=args.persona or None,
    )

    print(result["summary"])


if __name__ == "__main__":
    main()
