"""Execute complete candidate lists in author sample order."""
from .executors import ThreadPoolExecutor


def candidate_executor(existing, candidates):
    if existing is not None:return existing,False
    return ThreadPoolExecutor(max_workers=1),True


def execute_candidates(samples, parse, submit, deduplicate=False):
    if not isinstance(samples,list):
        raise ValueError('candidate generation must finish before environment execution')
    results=[];seen=set()
    for ordinal,raw in samples:
        action=parse(raw)
        if deduplicate and action in seen:continue
        seen.add(action)
        result=submit(ordinal,action).result()
        results.append((ordinal,action,result))
    return results
