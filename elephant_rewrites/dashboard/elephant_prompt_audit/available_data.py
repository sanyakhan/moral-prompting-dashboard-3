"""All saved source stories and scores for the existing dashboard experiments."""

from collections import Counter
import json

from .reader import CONTROL, NOV, RELEASE, extract_verdict, json_rows


SONNET_SOL = NOV / "paper_control_sonnet_sol_20261123"
OTHER_SOL = NOV / "paper_control_terra_opus_sol_212_20261123"


def full_pairs(cases):
    protocol = json.loads((OTHER_SOL / "experiment.json").read_text())
    if protocol["run"] != 1 or protocol["stance"] != "neutral":
        raise ValueError("Expected neutral run-one Sol control results.")
    pairs = {pid: {"pair_id": pid} for pid in cases}
    for row in pairs.values():
        for judge in ("opus", "sol"):
            for condition in ("original", "released_flip", "sonnet_flip", "terra_flip", "opus_flip"):
                row[f"{judge}_{condition}"] = ""
                row[f"{judge}_{condition}_raw"] = ""

    def save(pid, judge, condition, response, error=""):
        row = pairs[pid]
        row[f"{judge}_{condition}"] = extract_verdict(response, error)
        row[f"{judge}_{condition}_raw"] = response

    for judge in ("opus", "sol"):
        for row in json_rows(RELEASE / judge / "results.jsonl"):
            if row["model"] != judge or not row["custom_id"].endswith("_r1"):
                raise ValueError("Unexpected run or judge in the released-flip results.")
            condition = "original" if row["side"] == "original" else "released_flip"
            save(row["pair_id"], judge, condition, row["response"], row["error"])
    for row in json_rows(CONTROL / "results.jsonl"):
        if row["run"] == 1 and row["eligible"]:
            save(row["pair_id"], "opus", row["writer"] + "_flip", row["response"], row["judge_error"])

    requests = {row["custom_id"]: row for row in json_rows(SONNET_SOL / "requests.jsonl")}
    outputs = list(json_rows(SONNET_SOL / "batch_output.jsonl"))
    if len(outputs) != len(requests) or {row["custom_id"] for row in outputs} != requests.keys():
        raise ValueError("The Sonnet/Sol control outputs do not cover their saved requests.")
    for output in outputs:
        request = requests[output["custom_id"]]
        pid = request["pair_id"]
        if not request["custom_id"].endswith("_r1") or request["prompt"] != cases[pid].flips["sonnet"].text:
            raise ValueError("The full Sonnet/Sol input differs from the displayed rewrite.")
        body = (output.get("response") or {}).get("body") or {}
        choice = (body.get("choices") or [{}])[0]
        response = (choice.get("message") or {}).get("content") or ""
        save(pid, "sol", "sonnet_flip", response, output.get("error") or body.get("error"))
    for row in json_rows(OTHER_SOL / "sol/results.jsonl"):
        if row["run"] != 1 or not row["custom_id"].endswith("_r1"):
            raise ValueError("Expected run-one Sol control results.")
        save(row["pair_id"], "sol", row["writer"] + "_flip", row["response"], row["error"])
    return pairs


def triplet_metrics(posts_only, cases, case_ids=None):
    """Use all labeled triplets for each writer/judge, or an explicit common cohort."""
    selected = None if case_ids is None else set(case_ids)
    metrics = []
    union = set()
    for writer in ("sonnet", "terra"):
        for judge in ("opus", "sol"):
            labels = {}
            for row in posts_only["results"][judge]:
                if row["writer"] == writer and (selected is None or row["pair_id"] in selected):
                    labels[row["pair_id"], row["version"]] = row["label"]
            ids = sorted({pid for pid, _ in labels if all(
                labels.get((pid, version)) in {"NTA", "YTA"} for version in (1, 2, 3)
            )})
            union.update(ids)
            patterns = Counter("/".join(labels[pid, v] for v in (1, 2, 3)) for pid in ids)
            metrics.append({
                "writer": writer, "judge": judge, "dilemmas": len(ids), "included_ids": ids,
                "v1_v2_NTA_YTA": sum(labels[pid, 1] == "NTA" and labels[pid, 2] == "YTA" for pid in ids),
                "v1_v3_NTA_NTA": sum(labels[pid, 1] == labels[pid, 3] == "NTA" for pid in ids),
                "same_body": sum(cases[pid].same_body.get(writer, False) for pid in ids),
                "patterns": dict(patterns),
            })
    return {
        "metrics": metrics, "included_ids": sorted(union),
        "verdict_extraction": posts_only["summary"]["verdict_extraction"],
    }
