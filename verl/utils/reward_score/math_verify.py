from math_verify import parse, verify

def compute_score(solution_str, ground_truth) -> float:
  try:
    gold = parse(ground_truth)
    answer = parse(solution_str)
    # Order here is important!
    result = verify(gold, answer)
  except TimeoutError:
    print("TG_ERROR: TimeoutError in compute_score")
    result = False
  
  return result
