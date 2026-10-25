"""Apply the unchanged STAR boundary after a real sampled token."""
from icr_output_framing import stop_reason


def apply_boundary(seq,pending,decode,eos_ids,now,finished_status):
    tokens=list(seq.get_output_token_ids())
    if not tokens:raise RuntimeError('stop checking requires a sampled token')
    text=decode(tokens)
    reason=stop_reason(text,pending.request.get('stop_mode'),eos=tokens[-1] in eos_ids,
        now=now,submitted_at=pending.submitted,budget_seconds=pending.request.get('budget_seconds'))
    seq.output_text=text
    if reason is not None:
        pending.stop_reason=reason;seq.status=finished_status;seq.stop_reason='star_'+reason
    return reason
