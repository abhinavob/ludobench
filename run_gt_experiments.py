import argparse

from agents import HeuristicAgent, RandomAgent
from evaluation import evaluate
from game_theory_agent import GameTheoryAgent
from ludo_env import LudoEnv


def make_agent(name: str, depth: int):
    name = name.lower()
    if name == "gt":
        return GameTheoryAgent(search_depth=depth)
    if name == "heuristic":
        return HeuristicAgent()
    if name == "random":
        return RandomAgent()
    if name == "llm":
        # Lazy import so non-LLM runs do not require API setup.
        from llm_agent import LLMAgent
        try:
            from real_llm import real_llm
        except Exception as exc:
            raise RuntimeError(
                "Failed to load real LLM client. Set OPENAI_API_KEY and check real_llm.py config."
            ) from exc
        return LLMAgent(real_llm)
    if name == "dummy_llm":
        from llm_agent import LLMAgent
        from dummy_llm import dummy_llm
        return LLMAgent(dummy_llm)
    raise ValueError(f"Unknown agent: {name}")


def run_match(agent0_name: str, agent1_name: str, games: int, depth: int, seed_start: int):
    a0 = make_agent(agent0_name, depth)
    a1 = make_agent(agent1_name, depth)
    results = []

    for i in range(games):
        env = LudoEnv(seed=seed_start + i, num_players=2, player_ids=[0, 1])
        agents = {0: a0, 1: a1}
        env.play_game(agents)
        results.append(evaluate(env))

    wins = {0: 0, 1: 0}
    for r in results:
        wins[r["winner"]] += 1

    avg_steps = sum(r["steps"] for r in results) / len(results)
    avg_captures = sum(r["captures"] for r in results) / len(results)
    safe_move_ratio = sum(r["safe_moves"] / max(1, r["total_moves"]) for r in results) / len(results)
    total_fallbacks = sum(r.get("llm_fallbacks", 0) for r in results)
    total_llm_calls = sum(r.get("llm_calls", 0) for r in results)

    print(f"Match: P0={agent0_name} vs P1={agent1_name}")
    print(f"Games: {games}, Seed start: {seed_start}")
    print(f"Win rate P0: {wins[0] / games:.3f}")
    print(f"Win rate P1: {wins[1] / games:.3f}")
    print(f"Avg steps: {avg_steps:.2f}")
    print(f"Avg captures: {avg_captures:.2f}")
    print(f"Safe move ratio: {safe_move_ratio:.3f}")
    print(f"LLM fallback ratio (if LLM used): {total_fallbacks}/{max(1, total_llm_calls)}")


def main():
    parser = argparse.ArgumentParser(description="Run independent 2-player GameTheoryAgent benchmarks.")
    choices = ["gt", "heuristic", "random", "llm", "dummy_llm"]
    parser.add_argument("--p0", default="gt", choices=choices, help="Player 0 agent")
    parser.add_argument("--p1", default="heuristic", choices=choices, help="Player 1 agent")
    parser.add_argument("--games", type=int, default=50, help="Number of games")
    parser.add_argument("--depth", type=int, default=2, help="Expectiminimax search depth for GT agents")
    parser.add_argument("--seed-start", type=int, default=0, help="Starting random seed")
    args = parser.parse_args()

    run_match(args.p0, args.p1, args.games, args.depth, args.seed_start)


if __name__ == "__main__":
    main()
