"""Compare context and post-only judgments on the same frozen dilemma IDs."""

import hashlib
import json
import re

from .reader import HERE, NOV, POSTS_ONLY, TRIPLETS, json_rows


CONTEXT_FILES = {
    ("sonnet", "opus"): TRIPLETS["sonnet"] / "judge/results.jsonl",
    ("sonnet", "sol"): NOV / "three_versions213_sol_terra_20261122/sol/results.jsonl",
    ("terra", "opus"): TRIPLETS["terra"] / "judge/results.jsonl",
    ("terra", "sol"): TRIPLETS["terra"] / "sol/results.jsonl",
}


def context_revision():
    return tuple(path.stat().st_mtime_ns for path in CONTEXT_FILES.values())


def load_context_comparison(summary):
    ids = set(summary["included_ids"])
    if not ids:
        raise ValueError("The context comparison needs a nonempty matched cohort.")
    protocol = json.loads((POSTS_ONLY / "experiment.json").read_text())
    sources = protocol["source_sha256"]
    extraction = summary["verdict_extraction"]
    leading = re.compile(extraction["regex"], re.IGNORECASE)
    alternative = re.compile(extraction["reject_alternatives_regex"], re.IGNORECASE)
    table = []
    for current in summary["metrics"]:
        ids = set(current.get("included_ids", summary["included_ids"]))
        writer, judge = current["writer"], current["judge"]
        path = CONTEXT_FILES[writer, judge]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != sources[str(path.relative_to(HERE.parents[1]))]:
            raise ValueError("A historical context result changed after the post-only run was prepared.")
        rows = [row for row in json_rows(path)
                if row["pair_id"] in ids and row["custom_id"].endswith("_r1")]
        keys = {(row["pair_id"], row["version"]) for row in rows}
        if len(rows) != len(keys) or keys != {(pid, v) for pid in ids for v in (1, 2, 3)}:
            raise ValueError("Historical judgments must cover each matched version exactly once.")
        labels = {}
        for row in rows:
            if not row["prompt"].startswith(protocol["prefix"] + "Shared context:\n"):
                raise ValueError("Expected the historical shared-context condition.")
            response = row["raw_response"]
            match = leading.match(response)
            if row["error"] or not match or alternative.match(response[match.end():]):
                raise ValueError("A historical response has no unambiguous leading verdict.")
            labels[row["pair_id"], row["version"]] = match.group(1).upper()
        before12 = sum(labels[pid, 1] == "NTA" and labels[pid, 2] == "YTA" for pid in ids)
        before13 = sum(labels[pid, 1] == labels[pid, 3] == "NTA" for pid in ids)
        total = len(ids)
        if current["dilemmas"] != total:
            raise ValueError("Before and after must use the same denominator.")
        table.append({
            "Rewrite model": writer.title(),
            "Judge": "Opus 5.5" if judge == "opus" else "Sol",
            "Dilemmas": total,
            "V1 / V2 · NTA / YTA": f"{before12 / total:.1%} → {current['v1_v2_NTA_YTA'] / total:.1%}",
            "V1 / V3 · NTA / NTA": f"{before13 / total:.1%} → {current['v1_v3_NTA_NTA'] / total:.1%}",
        })
    return table
