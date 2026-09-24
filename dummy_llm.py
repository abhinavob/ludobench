import random

def dummy_llm(prompt):
    # For now, behave like a noisy heuristic
    import re
    legal = re.findall(r"'legal_moves': \[(.*?)\]", prompt)
    if legal:
        moves = list(map(int, legal[0].split(",")))
        return str(random.choice(moves))
    return "0"
