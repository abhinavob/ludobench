from ludo_env import LudoEnv
from agents import RandomAgent, HeuristicAgent
from llm_agent import LLMAgent
from dummy_llm import dummy_llm
from evaluation import evaluate
from real_llm import real_llm


def run(agent_list, player_ids=None, seeds=1):
    results = []

    for s in range(seeds):
        env = LudoEnv(seed=s, num_players=len(agent_list), player_ids=player_ids)
        agents = {pid: agent for pid, agent in zip(env.player_ids, agent_list)}
        env.play_game(agents)
        results.append(evaluate(env))

    return results


def summarize(results):
    n = len(results)
    total_fallbacks = sum(r.get("llm_fallbacks", 0) for r in results)
    total_llm_calls = sum(r.get("llm_calls", 0) for r in results)
    player_ids = results[0]["player_ids"] if results else []
    wins = {pid: 0 for pid in player_ids}
    for r in results:
        wins[r["winner"]] += 1
    return {
        "win_rate_by_player": {pid: wins[pid] / n for pid in wins},
        "avg_steps": sum(r["steps"] for r in results) / n,
        "avg_captures": sum(r["captures"] for r in results) / n,
        "safe_move_ratio": sum(
            r["safe_moves"] / max(1, r["total_moves"]) for r in results
        ) / n,
        "llm_fallback_ratio": f"{total_fallbacks}/{max(1, total_llm_calls)}"
    }


if __name__ == "__main__":
    agents = {
        "Random": RandomAgent(),
        "Heuristic": HeuristicAgent(),
        "LLM": LLMAgent(real_llm)
    }

    settings = [
        ("Random", "Random"),
        ("Random",None,"LLM",None),
        ("Random", "Random"),
        ("Random", "Random"),
        ("Heuristic", "Random"),
        ("Random","Heuristic"),
        ("Random", "LLM"),
        ("LLM", "Random"),
        ("LLM", "Heuristic"),
        ("Heuristic", "LLM"),
        ("LLM", "LLM"),
        ("Random", None, "Random", None),
        ("Random", None, "Heuristic", None),
        ("LLM", None, "Random", None),
        ("Heuristic", None, "LLM", None),
        ("Random", "Random", "Random"),
        ("Random", "Random", "Heuristic"),
        ("LLM", "Random", "Random"),
        ("LLM", "Heuristic", "Random"),
        ("Heuristic", "Random", "LLM"),
        ("Random", "Random", "Random", "Random"),
        ("LLM", "Random", "Random", "Random"),
        ("LLM", "Heuristic", "Random", "Random"),
        ("Heuristic", "LLM", "Random", "Random"),
        ("LLM", "LLM", "Heuristic", "Random")
    ]

    for matchup in settings:
        player_ids = [i for i, name in enumerate(matchup) if name is not None]
        agent_list = [agents[name] for name in matchup if name is not None]
        results = run(agent_list, player_ids=player_ids)
        summary = summarize(results)
        label = " vs ".join(name if name is not None else "-" for name in matchup)
        print(f"{label} → {summary}")
