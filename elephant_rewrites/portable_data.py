"""Portable, integrity-checked data access for the hosted read-only dashboard."""

from functools import lru_cache
import gzip
import hashlib
import json
from pathlib import Path

from dashboard.elephant_prompt_audit.reader import Dilemma, Prompt
from dashboard.elephant_prompt_audit.prompt_change import SelfFlip


HERE = Path(__file__).resolve().parent
DATA = HERE / 'data/dashboard.json.gz'


def revision():
    return (DATA.stat().st_mtime_ns,)


@lru_cache(maxsize=1)
def snapshot():
    content = DATA.read_bytes()
    manifest = json.loads((DATA.parent / 'manifest.json').read_text())
    if hashlib.sha256(content).hexdigest() != manifest['bundle_sha256']:
        raise ValueError('The dashboard snapshot differs from its verification manifest.')
    value = json.loads(gzip.decompress(content))
    if value['format_version'] != 1:
        raise ValueError('Unsupported dashboard snapshot version.')
    return value


def make_cases(rows):
    cases = {}
    for pid, row in rows.items():
        cases[pid] = Dilemma(
            original=Prompt(**row['original']), original_story=row['original_story'],
            flips={k: Prompt(**v) for k, v in row['flips'].items()},
            versions={k: tuple(Prompt(**p) for p in ps) for k, ps in row['versions'].items()},
            version_status=row['version_status'], same_body=row['same_body'],
        )
    return cases


def load_dashboard_data():
    value = snapshot()
    self_flips = {pid: SelfFlip(prompt=Prompt(**row['prompt']), target=row['target'],
                              sol_response=row['sol_response'], sol_raw_response=row['sol_raw_response'])
                  for pid, row in value['self_flips'].items()}
    return make_cases(value['cases']), value['pairs'], self_flips, value['changed_template'], value['posts_only']


def load_expanded_terra():
    value = snapshot()['terra_expansion']
    return {**value, 'cases': make_cases(value['cases'])}


def control_prompts():
    return tuple(snapshot()['control_prompts'])


def context_table(summary):
    ids = set(summary['included_ids'])
    for table in snapshot()['context_tables'].values():
        if ids == set(table['included_ids']):
            return table['rows']
    raise ValueError('No verified shared-context comparison exists for this source cohort.')
