"""Read the separately collected Yes/No response-format control."""

import hashlib
import json

from .reader import NOV, POSTS_ONLY, json_rows


YES_NO = NOV / "three_versions_yes_no_20261123"


def yes_no_revision():
    paths = [YES_NO / "completion_verification.json", YES_NO / "summary.json"]
    paths.extend(YES_NO / judge / "results.jsonl" for judge in ("opus", "sol"))
    return tuple(path.stat().st_mtime_ns if path.exists() else 0 for path in paths)


def load_yes_no():
    marker_path = YES_NO / "completion_verification.json"
    if not marker_path.exists():
        return None
    marker = json.loads(marker_path.read_text())
    protocol = json.loads((YES_NO / "experiment.json").read_text())
    if not marker["passed"] or protocol["run"] != 1 or protocol["stance"] != "neutral":
        raise ValueError("Expected verified neutral run-one Yes/No inputs.")
    summary_path = YES_NO / "summary.json"
    if hashlib.sha256(summary_path.read_bytes()).hexdigest() != marker["summary_sha256"]:
        raise ValueError("The Yes/No summary changed after verification.")
    results = {}
    for judge in marker["judges"]:
        path = YES_NO / judge / "results.jsonl"
        if hashlib.sha256(path.read_bytes()).hexdigest() != marker["result_sha256"][judge]:
            raise ValueError("The Yes/No results changed after verification.")
        results[judge] = list(json_rows(path))
        source = {row["custom_id"]: row for row in json_rows(POSTS_ONLY / judge / "results.jsonl")}
        for row in results[judge]:
            previous = source[row["source_custom_id"]]
            if (row["run"] != 1 or row["stance"] != "neutral" or row["judge"] != judge
                or row["story"] != previous["story"] or row["question"] != previous["question"]
                or row["prompt"] != protocol["prefix"] + previous["story"]):
                raise ValueError("A Yes/No input differs from the saved post-only story or question.")
    return {"results": results, "summary": json.loads(summary_path.read_text()), "prefix": protocol["prefix"]}


def format_comparison(rows):
    table = []
    for row in rows:
        n = row["dilemmas"]
        def change(before, after):
            return f"{row[before] / n:.1%} → {row[after] / n:.1%}" if n else "No parsed triplets"
        table.append({
            "Rewrite model": row["writer"].title(),
            "Judge": "Opus 5.5" if row["judge"] == "opus" else "Sol",
            "Dilemmas": n,
            "V1 / V2 · NTA/YTA → No/Yes": change("before_v1_v2_NTA_YTA", "after_v1_v2_No_Yes"),
            "V1 / V3 · NTA/NTA → No/No": change("before_v1_v3_NTA_NTA", "after_v1_v3_No_No"),
        })
    return table
