from math_verify import parse, verify

def compute_score(solution_str, ground_truth) -> float:
  gold = parse(ground_truth)
  answer = parse(solution_str)

  # Order here is important!
  return verify(gold, answer)
