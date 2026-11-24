"""Read the saved run-one Opus inputs and responses for the two matched experiments."""

import csv
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path


HERE = Path(__file__).resolve().parent
NOV = HERE.parents[1] / "artifacts" / "NOV"
CONTROL = NOV / "paper_flip_control250_20261122"
RELEASE = NOV / "released_flips250_opus_sol_20261123"
POSTS_ONLY = NOV / "three_versions_posts_only_20261123"
POSTS_ONLY_COMPLETE = POSTS_ONLY / "completion_verification.json"
TRIPLETS = {
    "sonnet": NOV / "three_versions_sonnet250_20261122",
    "terra": NOV / "terra213_sol_opus_verdicts_20261122",
}
FLIPS = ("released", "sonnet", "terra", "opus")
VERDICT = re.compile(r'''^\s*[*_`"“']{0,3}\s*(YTA|NTA)\b''', re.IGNORECASE)
ALTERNATIVE = re.compile(r'''^[*_`"”']*\s*(?:/|or\b|and\b)\s*[*_`"“']*(?:YTA|NTA)\b''', re.IGNORECASE)


def extract_verdict(text, error=""):
    match = VERDICT.match(text or "")
    if error or not match or ALTERNATIVE.match(text[match.end():]):
        return ""
    return match.group(1).upper()


def infer_question(text):
    """Identify a literal question span for cases without hand annotations."""
    marker = re.compile(r"\b(?:WIBTA|AITA|(?:am|are|is|was|would|will|did|do|should)\b[^\n?!]{0,100}\b(?:asshole|wrong|assholes|unfairly|unreasonable|failing|feel bad|asking for too much|fuck up))\b", re.I)
    candidates = []
    short = []
    for match in marker.finditer(text):
        end = text.find("?", match.start())
        if end >= 0 and end - match.start() <= 500 and "\n\n" not in text[match.start():end]:
            question = text[match.start():end + 1]
            if len(question) > 10:
                candidates.append(question)
            else:
                short.append(question)
    if candidates or short:
        return candidates[-1] if candidates else short[-1]
    title = re.search(r"(?im)^\s*((?:WIBTA|AITA)\b[^\n?]{0,400}?[.!])(?=\s|$)", text)
    return title.group(1) if title else ""


@dataclass(frozen=True)
class Prompt:
    text: str
    question: str
    speaker: str
    verdict: str
    raw_response: str = ""
    status: str = ""

    def __post_init__(self):
        if self.question and self.question not in self.text:
            raise ValueError(f"Crux does not match saved prompt: {self.question!r}")


@dataclass
class Dilemma:
    original: Prompt
    original_story: str
    flips: dict[str, Prompt]
    versions: dict[str, tuple[Prompt, ...]]
    version_status: dict[str, str] = field(default_factory=dict)
    same_body: dict[str, bool] = field(default_factory=dict)


def json_rows(path):
    with path.open(encoding="utf-8") as source:
        if path.suffix == ".json":
            yield from json.load(source)
        else:
            for line in source:
                if line.strip():
                    yield json.loads(line)


def load_control_prompts() -> tuple[str, str]:
    protocol = json.loads((CONTROL / "experiment.json").read_text(encoding="utf-8"))
    rewrite = (
        protocol["rewrite_prompt_before_story"]
        + "[original story]"
        + protocol["rewrite_prompt_after_story"]
    )
    return rewrite, protocol["judge_prefix"] + "[story]"


def posts_only_revision() -> tuple[int, ...]:
    paths = [POSTS_ONLY_COMPLETE, POSTS_ONLY / "summary.json"]
    paths.extend(POSTS_ONLY / judge / "results.jsonl" for judge in ("opus", "sol"))
    return tuple(path.stat().st_mtime_ns if path.exists() else 0 for path in paths)


def load_posts_only():
    """Use new verdicts only after both judges' outputs have been collected and verified."""
    if not POSTS_ONLY_COMPLETE.exists():
        return None
    marker = json.loads(POSTS_ONLY_COMPLETE.read_text())
    if not marker.get("embedded_writer_notes_removed"):
        return None
    protocol = json.loads((POSTS_ONLY / "experiment.json").read_text())
    if not marker["passed"] or marker["shared_context_in_judge_input"] is not False:
        raise ValueError("The post-only collection has not passed verification.")
    if protocol["run"] != 1 or protocol["stance"] != "neutral":
        raise ValueError("Expected neutral run-one inputs for the post-only collection.")
    summary_file = POSTS_ONLY / "summary.json"
    if hashlib.sha256(summary_file.read_bytes()).hexdigest() != marker["summary_sha256"]:
        raise ValueError("The post-only summary changed after verification.")
    results = {}
    for judge in ("opus", "sol"):
        path = POSTS_ONLY / judge / "results.jsonl"
        if hashlib.sha256(path.read_bytes()).hexdigest() != marker["result_sha256"][judge]:
            raise ValueError(f"The {judge} post-only verdicts changed after verification.")
        results[judge] = list(json_rows(path))
        for row in results[judge]:
            if row["run"] != 1 or row["stance"] != "neutral" or row["judge"] != judge:
                raise ValueError("Unexpected condition in the post-only verdicts.")
            if row["prompt"] != protocol["prefix"] + row["story"]:
                raise ValueError("A post-only judge input differs from the saved post.")
    return {"results": results, "summary": json.loads(summary_file.read_text())}


def load_cases(posts_only=None, *, include_all=False) -> dict[str, Dilemma]:
    if posts_only is None:
        posts_only = load_posts_only()
    matched = {
        r["pair_id"]
        for r in json_rows(TRIPLETS["terra"] / "judge/comparison_table.json")
        if r["body2_equals_body3"]
    }
    with (HERE / "speakers.tsv").open(encoding="utf-8", newline="") as source:
        speakers = {r["id"]: r for r in csv.DictReader(source, delimiter="\t")}
    questions = json.loads((HERE / "questions.json").read_text(encoding="utf-8"))
    if matched != speakers.keys() or matched != questions.keys():
        raise ValueError("The speaker and question annotations must cover the matched cohort.")

    release_protocol = json.loads((RELEASE / "experiment.json").read_text())
    if release_protocol["run"] != 1 or release_protocol["stance"] != "neutral":
        raise ValueError("Expected neutral run-one inputs for the released flips.")
    prefix = release_protocol["prefix"]
    release_labels = {
        (r["pair_id"], r["side"]): r["response"]
        for r in json_rows(RELEASE / "opus/results.jsonl")
        if r["model"] == "opus" and r["custom_id"].endswith("_r1")
    }
    original_inputs = {
        r["pair_id"]: r["prompt"]
        for r in json_rows(CONTROL / "original_judge/requests.jsonl")
        if r["run"] == 1
    }
    flip_inputs = {
        (r["pair_id"], r["writer"]): r["prompt"]
        for r in json_rows(CONTROL / "judge/requests.jsonl")
        if r["custom_id"].endswith("_r1")
    }
    flip_results = {
        (r["pair_id"], r["writer"]): r
        for r in json_rows(CONTROL / "results.jsonl")
        if r["run"] == 1
    }
    cases = {}
    for row in json_rows(CONTROL / "selected_pairs.json"):
        pid = row["pair_id"]
        if not include_all and pid not in matched:
            continue
        role, crux = speakers.get(pid, {}), questions.get(pid, {})
        original_raw = release_labels[pid, "original"]
        original = Prompt(original_inputs[pid], crux.get("original") or infer_question(row["original_story"]),
                          role.get("original", ""), extract_verdict(original_raw), raw_response=original_raw)
        released_raw = release_labels[pid, "flip"]
        flips = {
            "released": Prompt(prefix + row["existing_flip"], crux.get("released") or infer_question(row["existing_flip"]),
                               role.get("released") or role.get("other", ""),
                               extract_verdict(released_raw), raw_response=released_raw)
        }
        for writer in FLIPS[1:]:
            result = flip_results[pid, writer]
            if result["eligible"]:
                text = flip_inputs[pid, writer]
                flips[writer] = Prompt(text, crux.get(writer) or infer_question(text),
                                       role.get(writer) or role.get("other", ""),
                                       extract_verdict(result["response"], result["judge_error"]),
                                       raw_response=result["response"])
            elif result["issue"] == "N/A":
                flips[writer] = Prompt("", "", "", "", status="The writer returned N/A. No verdict was collected.")
            else:
                raise ValueError(f"Unexpected rewrite status for {pid}, {writer}: {result['issue']}")
        cases[pid] = Dilemma(original, row["original_story"], flips, {})

    for writer, folder in TRIPLETS.items():
        saved_judgments = (posts_only["results"]["opus"] if posts_only is not None
                           else json_rows(folder / "judge/results.jsonl"))
        judgments = {
            (r["pair_id"], r["version"]): r for r in saved_judgments
            if r["custom_id"].endswith("_r1") and r["pair_id"] in cases
            and (posts_only is None or r["writer"] == writer)
        }
        for row in json_rows(folder / "parsed_results.jsonl"):
            pid = row["pair_id"]
            if pid not in cases or not row["custom_id"].endswith("_r1"):
                continue
            cases[pid].version_status[writer] = row["status"]
            if row["status"] != "three_versions":
                cases[pid].versions[writer] = ()
                continue
            role = speakers.get(pid, {})
            versions = []
            for version in sorted(row["versions"], key=lambda v: v["version"]):
                number = version["version"]
                judge = judgments[pid, number]
                speaker = (role.get(f"{writer}_v1") or role.get("original", "")) if number == 1 else (
                    role.get(f"{writer}_v23") or role.get("other", "")
                )
                versions.append(Prompt(
                    judge["prompt"], version["question"], speaker,
                    judge["label"] if posts_only is not None else judge["raw_response"],
                    raw_response=judge["raw_response"],
                ))
            cases[pid].versions[writer] = tuple(versions)
            def body(prompt):
                before, question, after = prompt.text.rpartition(prompt.question)
                return (before + after).strip() if question else prompt.text.strip()
            cases[pid].same_body[writer] = body(versions[1]) == body(versions[2])
    return cases
