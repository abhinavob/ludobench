"""
What each decision offered and what was chosen, from the logged board, with
the repo's own definitions:
  groups       legal moves with identical outcomes (board_render.group_moves);
               a decision with fewer than 2 groups offers no real choice
  availability what each legal move does (decision_log.analyse), and the
               rate kinds of decision_log.RATE_KINDS: capture, safe (not a
               bring-out), open (a bring-out), home entry, finish
  search       the search player's move (GameTheoryMultiplayerAgent, depth 2,
               as run_spots logs it)
A spot record carries its groups, availability and search move; a game record
from before home entries were logged has no home_entry and no search move, so
those are computed here.
"""
from agents import HeuristicAgent
from board_render import group_moves
from decision_log import RATE_KINDS, analyse
from game_theory_multiplayer_agent import GameTheoryMultiplayerAgent
from spot_evaluation import SpotEnv

SEARCH = GameTheoryMultiplayerAgent(search_depth=2)
HEURISTIC = HeuristicAgent()


def env_of(d):
    return SpotEnv([int(p) for p in d["tokens_before"]],
                   {int(p): list(t) for p, t in d["tokens_before"].items()})


def complete(d):
    """d with groups, availability and search filled in where the record has none."""
    p, dice, legal = d["player_id"], d["dice"], d["legal_moves"]
    env = env_of(d)
    if not d.get("move_groups"):
        d["move_groups"] = group_moves(env, p, dice, legal)
    if not d.get("availability") or "home_entry" not in d["availability"]:
        d["availability"] = analyse(env, p, dice, HEURISTIC)["availability"]
    if d.get("search_move") is None:
        d["search_move"] = SEARCH.select_move(env, p, dice, legal)
    return d


def measures(d):
    """
    The per-decision quantities the tables average, for a completed decision:
    valid (a legal move from the model's own reply), choice (at least 2
    groups), agreement with the search player and its chance level under a
    uniformly random legal move, the first-listed move (by group and by piece)
    and its chance, and per rate kind: offered, taken, the random policy's
    rate and the search player's choice.
    """
    legal, groups, chosen, search = d["legal_moves"], d["move_groups"], d["chosen_index"], d["search_move"]
    group = {t: i for i, g in enumerate(groups) for t in g}
    size = lambda t: len(groups[group[t]]) / len(legal)
    m = {"valid": chosen is not None and not d["was_fallback"], "choice": len(groups) >= 2,
         "agree": chosen is not None and group[chosen] == group[search], "chance_agree": size(search)}
    shown = d.get("shown_legal_moves")
    if shown:
        first = shown[0]
        m.update(first=chosen is not None and group[chosen] == group[first], chance_first=size(first),
                 first_piece=chosen == first, chance_first_piece=1 / len(legal))
    for kind, tokens in RATE_KINDS.items():
        toks = tokens(d["availability"])
        m[kind] = {"offered": bool(toks), "taken": chosen in toks, "chance": len(toks) / len(legal),
                   "search": search in toks}
    return m
