def evaluate(env):
    moves = env.move_log

    captures = sum(m["capture"] for m in moves)
    safe_moves = sum(
        m["to"] in env.SAFE_SQUARES for m in moves if 0 <= m["to"] < env.BOARD_SIZE
    )

    return {
        "winner": env.winner,
        "num_players": env.NUM_PLAYERS,
        "player_ids": list(env.player_ids),
        "steps": len(moves),
        "captures": captures,
        "safe_moves": safe_moves,
        "total_moves": len(moves),
        "llm_fallbacks": env.llm_fallbacks,
        "llm_calls": env.llm_calls,
        "llm_fallback_ratio": f"{env.llm_fallbacks}/{max(1, env.llm_calls)}"
    }
