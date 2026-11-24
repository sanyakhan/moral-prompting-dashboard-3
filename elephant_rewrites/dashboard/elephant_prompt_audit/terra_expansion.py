"""Read only completed, verified Terra expansion outputs; never call providers."""

from collections import Counter
import hashlib
import json

from .reader import Dilemma, NOV, Prompt, infer_question, json_rows


EXPANSION = NOV / 'terra500_three_versions_yes_no_20261123'


def revision():
    files = [EXPANSION / name for name in ('completion_verification.json', 'summary.json', 'selected_triplets.jsonl')]
    files.extend(EXPANSION / judge / 'combined_results.jsonl' for judge in ('opus', 'sol'))
    return tuple(p.stat().st_mtime_ns if p.exists() else 0 for p in files)


def load_expansion():
    marker_file = EXPANSION / 'completion_verification.json'
    if not marker_file.exists():
        return None
    marker = json.loads(marker_file.read_text())
    protocol = json.loads((EXPANSION / 'experiment.json').read_text())
    if not marker['passed'] or protocol['run'] != 1 or protocol['stance'] != 'neutral':
        raise ValueError('Expected a completed neutral run-one Terra expansion.')
    if protocol['judge_prefix'] != 'Output only Yes or No.\n' or marker['shared_context_in_judge_input'] is not False:
        raise ValueError('The Terra expansion must use post-only Yes/No inputs.')

    def check(path, expected):
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Terra expansion file changed after verification: {path.name}')

    check(EXPANSION / 'summary.json', marker['summary_sha256'])
    check(EXPANSION / 'selected_triplets.jsonl', marker['selected_sha256'])
    summary = json.loads((EXPANSION / 'summary.json').read_text())
    selected = list(json_rows(EXPANSION / 'selected_triplets.jsonl'))
    ids = {r['pair_id'] for r in selected}
    if len(ids) != len(selected) or len(ids) != protocol['target_usable_triplets']:
        raise ValueError('The Terra expansion has missing or repeated source IDs.')
    scores = {}
    for judge in ('opus', 'sol'):
        path = EXPANSION / judge / 'combined_results.jsonl'
        check(path, marker['result_sha256'][judge])
        rows = list(json_rows(path))
        scores[judge] = {(r['pair_id'], r['version']): r for r in rows}
        if len(rows) != len(scores[judge]) or len(rows) != len(ids)*3:
            raise ValueError('Terra expansion verdicts do not cover each triplet exactly once.')
        for row in rows:
            if (row['pair_id'] not in ids or row['run'] != 1 or row['stance'] != 'neutral'
                    or row['writer'] != 'terra' or row['judge'] != judge
                    or row['prompt'] != protocol['judge_prefix'] + row['story']):
                raise ValueError('Unexpected Terra expansion scoring condition.')

    cases = {}
    for row in selected:
        pid = row['pair_id']
        versions = []
        for v in row['versions']:
            a, b = (scores[judge][pid, v['version']] for judge in ('opus', 'sol'))
            if a['story'] != b['story'] or a['story'] != v['story'] or a['question'] != v['question']:
                raise ValueError('Terra expansion judges received different posts.')
            versions.append(Prompt(a['prompt'], v['question'], '', a['label'], raw_response=a['raw_response']))
        if row['versions'][1]['body'] != row['versions'][2]['body']:
            raise ValueError('V2 and V3 differ in the controlled Terra expansion.')
        cases[pid] = Dilemma(
            original=Prompt(row['original_story'], infer_question(row['original_story']), '', '',
                            status='Reference original. Yes/No scores are for the rewritten V1-V3 posts.'),
            original_story=row['original_story'], flips={}, versions={'terra': tuple(versions)},
            version_status={'terra': 'three_versions'}, same_body={'terra': True},
        )
    return {'cases': cases, 'results': {j: list(rows.values()) for j, rows in scores.items()},
            'summary': summary, 'prefix': protocol['judge_prefix']}


def main_metrics(expansion):
    return [{**row, 'after_v1_v2_No_Yes': row['v1_v2_No_Yes'], 'after_v1_v3_No_No': row['v1_v3_No_No']}
            for row in expansion['summary']['metrics'] if row['scope'] == 'all_selected500']


def cohort_table(expansion):
    output = []
    names = {'reused': 'Previously scored cases', 'new_extension': 'New cases'}
    for row in expansion['summary']['metrics']:
        if row['scope'] not in names:
            continue
        n = row['dilemmas']
        rate = lambda key: f"{row[key]/n:.1%} ({row[key]}/{n})" if n else 'No parsed triplets'
        output.append({'Cases': names[row['scope']], 'Judge': 'Opus 5.5' if row['judge'] == 'opus' else 'Sol',
                       'V1 / V2 · No / Yes': rate('v1_v2_No_Yes'), 'V1 / V3 · No / No': rate('v1_v3_No_No')})
    return output
