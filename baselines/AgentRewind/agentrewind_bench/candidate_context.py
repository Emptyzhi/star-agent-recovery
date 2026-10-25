"""Lossless references for repeated controller candidate receipts at render time."""
from copy import deepcopy
import json
import re


CATALOG_PREFIX = '[AGENTREWIND_CANDIDATE_HISTORY]\n'
FEEDBACK_MARKER = '[AGENTREWIND_TOOL_FEEDBACK]\n'
REFERENCE_KEY = 'candidate_history_refs'


def candidate_receipts(messages):
    receipts = []
    decoder = json.JSONDecoder()
    for index, message in enumerate(messages):
        if message['role'] != 'user':
            continue
        text = message['content']
        starts = [0] + [match.end() for match in re.finditer(re.escape(FEEDBACK_MARKER), text)]
        for start in starts:
            fragment = text[start:]
            if not fragment.startswith('{'):
                continue
            # Controller receipts are JSON; native observations need not be JSON.
            try:
                value, length = decoder.raw_decode(fragment)
            except json.JSONDecodeError:
                if start:
                    raise ValueError('controller feedback contains invalid JSON')
                continue
            if type(value) is not dict or value.get('tool') != 'backtrack_candidates' or 'result' not in value:
                continue
            result = value['result']
            if type(result) is list:
                if any(type(entry) is not dict or type(entry.get('record_uid')) is not str for entry in result):
                    raise ValueError('candidate receipt entries require literal record_uid objects')
            elif type(result) is not dict or set(result) != {REFERENCE_KEY}:
                raise ValueError('unexpected candidate receipt result')
            receipts.append(dict(message=index, start=start, end=start+length, value=value))
    return receipts


def _key(entry):
    return json.dumps(entry, sort_keys=True, ensure_ascii=False, allow_nan=False)


def _ranges(references):
    ranges = []
    for source, index in references:
        if ranges and ranges[-1][0] == source and ranges[-1][2] == index:
            ranges[-1][2] += 1
        else:
            ranges.append([source, index, index+1])
    return ranges


def project_candidate_history(messages):
    if any(m['role'] == 'system' and m['content'].startswith(CATALOG_PREFIX) for m in messages):
        raise ValueError('candidate history projection must start from original logical records')
    receipts = candidate_receipts(messages)
    if any(type(receipt['value']['result']) is not list for receipt in receipts):
        raise ValueError('candidate references require their matching catalog')
    report = dict(receipts=len(receipts), referenced_receipts=0, historical_unique_entries=0,
                  referenced_entries=0)
    if len(receipts) < 2:
        return messages, report
    latest = receipts[-1]['value']['result']
    lookup = {}
    for index, entry in enumerate(latest):
        lookup.setdefault(_key(entry), ['latest', index])
    historical = []
    replacements = {}
    for receipt in receipts[:-1]:
        refs = []
        for entry in receipt['value']['result']:
            key = _key(entry)
            if key not in lookup:
                lookup[key] = ['historical', len(historical)]
                historical.append(entry)
            refs.append(lookup[key])
        value = dict(receipt['value'], result={REFERENCE_KEY: _ranges(refs)})
        replacements.setdefault(receipt['message'], []).append((receipt, value))
        report['referenced_entries'] += len(refs)
    projected = deepcopy(messages)
    for index, changes in replacements.items():
        text = projected[index]['content']
        for receipt, value in reversed(changes):
            text = text[:receipt['start']] + json.dumps(value, ensure_ascii=False) + text[receipt['end']:]
        projected[index]['content'] = text
    catalog = dict(latest_receipt_ordinal=len(receipts)-1, historical_entries=historical)
    explanation = (
        '\nEarlier controller candidate receipts use candidate_history_refs. Each [source,start,end] '
        'is the exact zero-based slice [start,end), in order, from the latest full candidate result '
        '(source=latest) or historical_entries above (source=historical). Reference order '
        'preserves the earlier list order and membership. Distinct historical versions '
        'remain literal. These references change presentation only. Select commit targets '
        'from the latest returned candidate list, not historical entries. Raw model text '
        'and native observations are unchanged.')
    first_user = next(i for i, m in enumerate(projected) if m['role'] == 'user')
    projected.insert(first_user, dict(role='system', content=CATALOG_PREFIX+
        json.dumps(catalog, ensure_ascii=False)+'\n'+explanation))
    report.update(referenced_receipts=len(receipts)-1, historical_unique_entries=len(historical))
    return projected, report


def expand_candidate_history(messages):
    """Reconstruct every original candidate receipt for independent round-trip checks."""
    catalogs = [m for m in messages if m['role']=='system' and m['content'].startswith(CATALOG_PREFIX)]
    if not catalogs:
        return deepcopy(messages)
    if len(catalogs) != 1:
        raise ValueError('candidate history requires exactly one catalog')
    catalog = json.JSONDecoder().raw_decode(catalogs[0]['content'][len(CATALOG_PREFIX):])[0]
    expanded = deepcopy([m for m in messages if m is not catalogs[0]])
    receipts = candidate_receipts(expanded)
    if catalog['latest_receipt_ordinal'] != len(receipts)-1:
        raise ValueError('candidate history latest receipt does not match')
    latest = receipts[-1]['value']['result']
    if type(latest) is not list:
        raise ValueError('latest candidate result must remain complete')
    sources = dict(latest=latest, historical=catalog['historical_entries'])
    for receipt in reversed(receipts[:-1]):
        value = receipt['value']
        refs = value['result']
        if type(refs) is not dict or set(refs) != {REFERENCE_KEY}:
            raise ValueError('old candidate receipt must contain explicit references')
        entries = []
        for source, start, end in refs[REFERENCE_KEY]:
            if (source not in sources or type(start) is not int or type(end) is not int
                    or not 0 <= start < end <= len(sources[source])):
                raise ValueError('invalid candidate history reference')
            entries.extend(deepcopy(sources[source][start:end]))
        value = dict(value, result=entries)
        text = expanded[receipt['message']]['content']
        expanded[receipt['message']]['content'] = (text[:receipt['start']]+
            json.dumps(value, ensure_ascii=False)+text[receipt['end']:])
    return expanded
