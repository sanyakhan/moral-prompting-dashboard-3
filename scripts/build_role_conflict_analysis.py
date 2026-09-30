from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from moral_protocols.elephant_syc import parse_aita_label


REPO = Path(__file__).resolve().parents[2]
RUN = REPO / "artifacts/role_conflict/full250_ab_20260928"
OUTPUT = REPO / "dashboard3_deploy/output/analysis"
MODELS = {
    "Opus 5": (
        "anthropic_responses.csv",
        ["anthropic_patch_responses.csv", "anthropic_retry2_responses.csv"],
    ),
    "GPT-5.6-sol": ("openai_responses.csv", []),
    "Gemini 3.7 Flash": (
        "google_responses.csv",
        ["google_patch_responses.csv", "google_retry2_responses.csv"],
    ),
    "DeepSeek V4 Flash": ("deepseek_responses.csv", ["deepseek_patch_responses.csv"]),
    "GLM-5.3 Flash": ("zai_responses_dedup.csv", ["zai_patch_responses.csv"]),
}
PLACEMENTS = {
    "": ("U", "all_user"),
    "_s_framing": ("F", "system_framing"),
    "_s_framing_output": ("F+O", "system_framing_output"),
}


def merge_responses(base_name: str, patch_names: list[str]) -> pd.DataFrame:
    data = pd.read_csv(RUN / base_name)
    for patch_name in patch_names:
        patch = pd.read_csv(RUN / patch_name)
        patch["label"] = patch["response_raw"].map(parse_aita_label)
        replacements = (
            patch.loc[patch["label"].isin(["A", "B"])]
            .drop_duplicates("custom_id", keep="last")
            .set_index("custom_id")["response_raw"]
        )
        data["response_raw"] = data["custom_id"].map(replacements).fillna(
            data["response_raw"]
        )
    return data


def condition_parts(condition: str) -> tuple[str, str, str, str, str]:
    for suffix, (short, placement) in sorted(
        PLACEMENTS.items(), key=lambda item: len(item[0]), reverse=True
    ):
        if suffix and condition.endswith(suffix):
            base = condition.removesuffix(suffix)
            break
    else:
        short, placement, base = "U", "all_user", condition
    target, owner = base.split("_", 1)
    return base, target, owner, short, placement


def score_model(model: str, data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["label"] = data["response_raw"].map(parse_aita_label)
    data["parsed"] = data["label"].isin(["A", "B"])
    data["score"] = data["parsed"] & data["label"].eq(data["urgency_correct_letter"])
    parts = data["condition"].map(condition_parts).apply(pd.Series)
    parts.columns = ["condition_base", "target", "owner", "placement_short", "placement"]
    for column in parts:
        data[column] = parts[column]
    data["model"] = model
    data["scenario"] = data["sample_id"].str.rsplit("_", n=1).str[-1].astype(int)
    return data


def bootstrap_summary(rows: pd.DataFrame, rng: np.random.Generator) -> tuple:
    parsed = rows.loc[rows["parsed"]]
    estimate = 100 * parsed["score"].mean()
    by_scenario = rows.groupby("scenario").agg(successes=("score", "sum"), parsed=("parsed", "sum"))
    samples = rng.integers(0, len(by_scenario), size=(2000, len(by_scenario)))
    boot = 100 * by_scenario["successes"].to_numpy()[samples].sum(axis=1) / by_scenario["parsed"].to_numpy()[samples].sum(axis=1)
    low, high = np.quantile(boot, [0.025, 0.975])
    return estimate, low, high, len(rows) - len(parsed), len(rows)


def summary_row(model: str, view: str, group: str, rows: pd.DataFrame, rng: np.random.Generator) -> dict:
    estimate, low, high, unparsed, eligible = bootstrap_summary(rows, rng)
    return {"dataset": "Role Conflict", "view": view, "group": group, "model": model, "estimate": estimate, "ci_low": low, "ci_high": high, "unparsed": unparsed, "eligible": eligible, "parsed": eligible - unparsed}


def build() -> None:
    rng = np.random.default_rng(20260930)
    summaries, cells, directions = [], [], []
    for model, (base, patches) in MODELS.items():
        scored = score_model(model, merge_responses(base, patches))
        summaries.append(summary_row(model, "model_overall", "overall", scored, rng))
        for owner, rows in scored.groupby("owner", sort=True):
            summaries.append(summary_row(model, "decision_owner_all_placements", owner, rows, rng))
        for target, rows in scored.groupby("target", sort=True):
            summaries.append(summary_row(model, "target_actor_all_placements", target, rows, rng))
        for placement, rows in scored.groupby("placement_short", sort=True):
            summaries.append(summary_row(model, "placement", placement, rows, rng))
        for (base_condition, placement), rows in scored.groupby(["condition_base", "placement"], sort=True):
            parsed = rows.loc[rows["parsed"]]
            cells.append({"model": model, "condition_base": base_condition, "placement": placement, "urgency_correct_percent": 100 * parsed["score"].mean(), "parsed": len(parsed), "unparsed": len(rows) - len(parsed)})

        placement_values = {}
        for placement in ["all_user", "system_framing", "system_framing_output"]:
            selected = scored.loc[scored["placement"].eq(placement) & scored["condition_base"].isin(["user_sdc", "sdc_user"]) & scored["parsed"]]
            means = selected.groupby("condition_base")["score"].mean()
            placement_values[placement] = 100 * (means["user_sdc"] - means["sdc_user"])
        selected = scored.loc[scored["condition_base"].isin(["user_sdc", "sdc_user"]) & scored["parsed"]]
        pivot = selected.pivot_table(index=["scenario", "placement"], columns="condition_base", values="score", aggfunc="mean").dropna()
        scenario_diff = (100 * (pivot["user_sdc"] - pivot["sdc_user"])).groupby("scenario").mean()
        samples = rng.integers(0, len(scenario_diff), size=(2000, len(scenario_diff)))
        boot = scenario_diff.to_numpy()[samples].mean(axis=1)
        directions.append({"model": model, "estimate": scenario_diff.mean(), "ci_low": np.quantile(boot, .025), "ci_high": np.quantile(boot, .975), **placement_values})

    OUTPUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).to_csv(OUTPUT / "role_conflict_collapsed_summaries.csv", index=False)
    pd.DataFrame(cells).to_csv(OUTPUT / "role_conflict_condition_cells.csv", index=False)
    pd.DataFrame(directions).to_csv(OUTPUT / "role_conflict_role_direction.csv", index=False)
    (OUTPUT / "role_conflict_metadata.json").write_text(json.dumps({"models": list(MODELS), "bootstrap_iterations": 2000, "seed": 20260930, "source": str(RUN.relative_to(REPO))}, indent=2))


if __name__ == "__main__":
    build()
