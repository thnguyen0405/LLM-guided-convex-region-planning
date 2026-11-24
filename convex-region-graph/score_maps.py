"""Prepare class-1 data and score query priority with local Ollama.

Python prepares features and class-1 labels. The LLM directly returns priority scores in [0,1]. Python validates and saves
the responses unchanged. Old scoring artifacts remain untouched.
"""
import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from urllib.request import Request, urlopen

from prepare_scoring_data import ROOT, DEFAULT_CONFIG, prepare_file, digest

MODEL = 'qwen3:4b-instruct'
DEFAULT_PROMPT = ROOT / 'query_scoring_prompt.txt'
DEFAULT_RUBRIC = ROOT / 'query_priority_rubric.json'
LABEL_NAMES = tuple(json.loads(DEFAULT_RUBRIC.read_text())['label_names'])
SCHEMA = {
    'type': 'object',
    'properties': {
        'status': {'type': 'string', 'enum': ['scored', 'insufficient_data']},
        'score': {'anyOf': [{'type': 'number', 'minimum': 0, 'maximum': 1}, {'type': 'null'}]},
        'reason': {'type': 'string', 'minLength': 1, 'maxLength': 320},
    },
    'required': ['status', 'score', 'reason'], 'additionalProperties': False,
}
def unit_number(value):
    return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1


def validate_reason(reason):
    if not isinstance(reason, str) or not reason.strip() or len(reason) > 320:
        raise ValueError('Explanation must contain 1 to 320 characters')


def validate_answer(answer):
    if not isinstance(answer, dict) or set(answer) != set(SCHEMA['required']):
        raise ValueError('Response must contain exactly status, score and reason')
    if answer['status'] not in ('scored', 'insufficient_data'):
        raise ValueError('Unknown scoring status')
    validate_reason(answer['reason'])
    score = answer['score']
    if answer['status'] == 'scored':
        if not unit_number(score):
            raise ValueError('Score must be a finite number from 0 to 1')
    elif score is not None:
        raise ValueError('Insufficient data requires a null score')
    return answer


def ask_ollama(task, features, *, model=MODEL, prompt=None, base_url='http://localhost:11434', num_ctx=32768):
    payload = {'model': model, 'messages': [
        {'role': 'system', 'content': prompt if prompt is not None else DEFAULT_PROMPT.read_text()},
        # Keep common graph context first so Ollama can reuse its prompt cache.
        {'role': 'user', 'content': json.dumps({**features, 'task': task}, allow_nan=False, separators=(',', ':'))}],
        'format': SCHEMA, 'stream': False,
        'options': {'temperature': 0, 'num_predict': 180,
                    'num_ctx': num_ctx}}
    request = Request(base_url.rstrip('/') + '/api/chat',
                      data=json.dumps(payload, allow_nan=False).encode(),
                      headers={'Content-Type': 'application/json'})
    with urlopen(request, timeout=600) as response:
        result = json.load(response)
    if result.get('done_reason') == 'length':
        raise ValueError('Ollama response truncated at the output-token limit; no score was accepted')
    if 'prompt_eval_count' in result:
        count = result['prompt_eval_count']
        print(f"  Ollama prompt tokens: {count}", flush=True)
        if count >= payload['options']['num_ctx'] - payload['options']['num_predict']:
            raise ValueError('Graph prompt may exceed the context budget; no score was accepted')
    return json.loads(result['message']['content'])


def context_for(data):
    evidence = {}
    for name in LABEL_NAMES:
        memberships = [item['memberships'][name] for item in data['regions'] + data['connections']
                       if name in item['memberships']]
        known = [m['value'] for m in memberships if m['value'] is not None]
        evidence[name] = {'available_count': len(known),
                          'active_count': sum(m['active'] is True for m in memberships),
                          'unavailable_count': len(memberships) - len(known),
                          'membership_min': min(known) if known else None,
                          'membership_max': max(known) if known else None,
                          'missing_reasons': sorted({m['reason'] for m in memberships
                                                     if m['value'] is None})}
    return {'map': data['map'], 'planning': data['planning'], 'query': data['query'],
            'label_evidence': evidence,
            'sequences': data['sequences'],
            'retained_adjacency': [
                {'source': direction['source'], 'target': direction['target'], 'weight': direction['weight']}
                for edge in data['connections'] for direction in edge['directions']],
            'graph': {
                'nodes': [{key: region[key] for key in
                           ('id', 'centroid', 'structural_features', 'labels')} | {
                               'memberships': {name: m['value'] for name, m in region['memberships'].items()}}
                          for region in data['regions']],
                'connections': [{key: edge[key] for key in
                                 ('source', 'target', 'safe_portal', 'weight', 'structural_features', 'labels')} | {
                                     'memberships': {name: m['value'] for name, m in edge['memberships'].items()}}
                                for edge in data['connections']]},
            'class1_label_rules': data['configuration']['label_rules'],
            'feature_definitions': data['configuration']['feature_definitions'],
            'pending_agreement': data['configuration']['pending_agreement'],
            'limitations': data['limitations']}


def atomic_save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def explanation_review_flags(answer, item, connection=False):
    """Flag possible unsupported label claims without altering model scores.

    This conservative text check is a review aid, not semantic verification.
    Python-computed labels remain the only authoritative labels.
    """
    reason = answer['reason'].lower()
    flags = []
    terms = {'junction': r'\bjunction\b', 'corridor': r'\bcorridor\b',
             'open': r'\bopen\b', 'dead_end': r'\bdead[- ]end\b',
             'pocket': r'\bpocket\b', 'narrow': r'\bnarrow\b',
             'critical': r'\bcritical\b', 'alternative_route': r'\balternative route\b'}
    terms.update({'portal_1d': r'\bportal[_ -]1d\b|\b1d portal\b',
                  'short_safe_portal': r'\bshort[_ -]safe[_ -]portal\b'})
    for label, pattern in terms.items():
        for match in re.finditer(pattern, reason):
            prefix = reason[max(0, match.start()-45):match.start()]
            if re.search(r'\b(no|not|without|inactive|unavailable|missing)\b', prefix):
                continue
            evidence = item['memberships'].get(label)
            if evidence is None or evidence['active'] is not True:
                flags.append(f'Possible unsupported positive label claim: {label}')
                break
    if connection and not item['structural_features']['mandatory_on_all_start_goal_paths']:
        for match in re.finditer(r'\b(mandatory|unavoidable|required on all|every path)\b', reason):
            prefix = reason[max(0, match.start()-30):match.start()]
            if not re.search(r'\b(no|not|without)\b', prefix):
                flags.append('Possible all-path necessity claim not supported by the graph test')
                break
    return flags


def score_prepared(data, *, model=MODEL, prompt_path=DEFAULT_PROMPT,
                   output, resume=False, base_url='http://localhost:11434', request_fn=None,
                   rubric_path=DEFAULT_RUBRIC, num_ctx=32768):
    if type(num_ctx) is not int or num_ctx <= 180:
        raise ValueError('num_ctx must be an integer greater than the 180-token output budget')
    prompt_path = Path(prompt_path)
    prompt = prompt_path.read_text(encoding='utf-8')
    rubric = json.loads(Path(rubric_path).read_text(encoding='utf-8'))
    if (rubric.get('score_source') != 'llm_direct_query_priority'
            or rubric.get('range') != [0, 1] or rubric.get('label_names') != list(LABEL_NAMES)):
        raise ValueError('Unsupported rubric score source, range or label vocabulary')
    fingerprint = hashlib.sha256(json.dumps(data, sort_keys=True, allow_nan=False).encode()).hexdigest()
    provenance = {'prepared_data_sha256': fingerprint, 'prompt_sha256': digest(prompt_path),
                  'model': model, 'temperature': 0, 'base_url': base_url,
                  'response_schema': SCHEMA,
                  'rubric_sha256': digest(rubric_path), 'num_predict': 180,
                  'num_ctx': num_ctx,
                  'request_format': 'context_first_direct_priority_only/v1'}
    result = {'schema': 'query-priority-scores/v4', 'map': data['map'], 'model': model,
              'score_meaning': 'query_specific_priority_not_local_geometric_suitability',
              'score_range': [0, 1], 'score_source': 'llm_direct_query_priority',
              'rubric': rubric,
              'run': provenance, 'status': 'in_progress',
              'limitations': data['limitations'] + data['configuration']['pending_agreement'],
              'regions': [], 'edges': []}
    output = Path(output)
    if output.exists() and json.loads(output.read_text()).get('schema') != 'query-priority-scores/v4':
        raise ValueError('Old score artifact protected; use a new output path for [0,1] scoring')
    if resume and output.exists():
        result = json.loads(output.read_text())
        if result.get('schema') != 'query-priority-scores/v4' or result.get('run') != provenance:
            raise ValueError('Cannot resume: model, input data, server, or prompt changed')
    context = context_for(data)
    context['scoring_rubric'] = rubric
    by_id = {r['id']: r for r in data['regions']}
    tasks = [('regions', r['id'], 'query_region_priority', {'id': r['id']}, r)
             for r in data['regions']]
    tasks += [('edges', (e['source'], e['target']), 'query_connection_priority',
               {'source': e['source'], 'target': e['target']},
               {**e, 'endpoint_regions': [by_id[e['source']], by_id[e['target']]]})
              for e in data['connections']]
    call = request_fn or ask_ollama
    result['status'] = 'in_progress'
    for collection, key, task, identity, item in tasks:
        def identity_key(record):
            return record['id'] if collection == 'regions' else (record['source'], record['target'])
        existing = [record for record in result[collection] if identity_key(record) == key]
        if existing:
            if len(existing) != 1:
                raise ValueError('Duplicate checkpoint item')
            validate_answer({k: existing[0][k] for k in SCHEMA['required']})
            continue
        print(f"{data['map']}: {task} {key}", flush=True)
        if data['query']['status'] != 'connected':
            answer = {'status': 'insufficient_data', 'score': None,
                      'reason': 'No start-goal sequence in the supplied query graph; priority is undefined.'}
        else:
            answer = validate_answer(call(task, {'context': context, 'item': item},
                                          model=model, prompt=prompt, base_url=base_url, num_ctx=num_ctx))
        result[collection].append({**identity, **answer})
        atomic_save(output, result)
    result['status'] = 'complete'
    targets = {r['id']: r for r in data['regions']}
    connections = {(e['source'], e['target']): e for e in data['connections']}
    for record in result['regions']:
        record['review_flags'] = explanation_review_flags(record, targets[record['id']])
    for record in result['edges']:
        record['review_flags'] = explanation_review_flags(
            record, connections[record['source'], record['target']], connection=True)
    result['review'] = {
        'calibration': 'provisional_requires_supervisor_agreement',
        'flagged_items': sum(bool(r['review_flags']) for r in result['regions'] + result['edges']),
        'note': 'Heuristic text flags require human review; they do not prove an error, '
                'change scores, modify labels, or reject graph elements.'}
    atomic_save(output, result)
    return result


def score_map(map_name, *, graph=None, config=DEFAULT_CONFIG, k=None,
              prepared_output=None, output=None, prompt=DEFAULT_PROMPT,
              model=MODEL, resume=False, prepare_only=False, base_url='http://localhost:11434',
              rubric=DEFAULT_RUBRIC, num_ctx=32768):
    graph = Path(graph or ROOT / 'research/data' / map_name / 'graph.json')
    prepared_output = Path(prepared_output or ROOT / 'outputs' / map_name / 'query_labeled_data_llm_01.json')
    output = Path(output or ROOT / 'outputs' / map_name / 'query_priority_llm_01.json')
    protected = {graph.resolve(), Path(config).resolve(), Path(prompt).resolve(), Path(rubric).resolve()}
    protected.update((ROOT / 'outputs' / map_name / name).resolve() for name in
                     ('scores.json', 'query_priority_scores.json', 'query_priority_scores_portal_1d.json',
                      'query_priority_weights_01.json', 'query_priority_weights_01_initial_review.json'))
    if prepared_output.resolve() in protected or output.resolve() in protected or output.resolve() == prepared_output.resolve():
        raise ValueError('New artifacts must not overwrite graph, config, prompt, legacy scores, or each other')
    data = prepare_file(graph, config, k, map_name, prepared_output)
    print(f"Prepared {data['query']['returned_k']} sequences; saved {prepared_output}", flush=True)
    if prepare_only:
        return data
    return score_prepared(data, model=model, prompt_path=prompt, output=output,
                          resume=resume, base_url=base_url, rubric_path=rubric, num_ctx=num_ctx)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--map', required=True, choices=['hole', 'narrow', 'flappy', 'room'])
    parser.add_argument('--graph', type=Path, help='Explicit graph path; defaults to the prepared research graph')
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--k', type=int)
    parser.add_argument('--prepared-output', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--prompt', type=Path, default=DEFAULT_PROMPT)
    parser.add_argument('--rubric', type=Path, default=DEFAULT_RUBRIC)
    parser.add_argument('--model', default=MODEL)
    parser.add_argument('--base-url', default='http://localhost:11434')
    parser.add_argument('--num-ctx', type=int, default=32768, help='Ollama context size; larger graphs may need more memory')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    score_map(args.map, graph=args.graph, config=args.config, k=args.k,
              prepared_output=args.prepared_output, output=args.output, prompt=args.prompt,
              model=args.model, resume=args.resume, prepare_only=args.prepare_only,
              base_url=args.base_url, rubric=args.rubric, num_ctx=args.num_ctx)


if __name__ == '__main__':
    main()
