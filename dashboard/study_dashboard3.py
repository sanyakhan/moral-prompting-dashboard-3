from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots


ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = ROOT / "output" / "images"
COLLAPSED_RESULTS_PATH = ROOT / "output" / "analysis" / "bootstrap_collapsed_results_20260919.json"
DEO_CELL_PATH = ROOT / "output" / "analysis" / "iteration2_figure1_condition_cells.csv"
ELEPHANT_CELL_PATH = ROOT / "output" / "analysis" / "iteration2_elephant_condition_cells.csv"
ELEPHANT_ROLE_PATH = ROOT / "output" / "analysis" / "iteration2_elephant_role_direction.csv"
DEO_PARTS_DIR = ROOT / "prompts" / "deo_consq" / "parts"
ELEPHANT_PROMPT_DIR = ROOT / "prompts" / "elephant_syc"
AIRISK_ALL_USER = (
    ROOT
    / "artifacts"
    / "airisk"
    / "owner_wording_all_user_6x250_20260920"
    / "prompts_ready.csv"
)
AIRISK_SYSTEM = (
    ROOT
    / "artifacts"
    / "airisk"
    / "original_system_placements_6x250_20260920"
    / "prompts_ready.csv"
)

MODEL_COLORS = {
    "Opus 5": "#4263b8",
    "GPT-5.6-sol": "#218c74",
    "Gemini 3.7 Flash": "#2296df",
    "DeepSeek V4 Flash": "#9b51e0",
    "GLM-5.3 Flash": "#e84393",
}
MODEL_TICK_LABELS = {
    "Opus 5": "Opus 5",
    "GPT-5.6-sol": "GPT-5.6-sol",
    "Gemini 3.7 Flash": "Gemini 3.7<br>Flash",
    "DeepSeek V4 Flash": "DeepSeek V4<br>Flash",
    "GLM-5.3 Flash": "GLM-5.3<br>Flash",
}

PLACEMENT_LABELS = {
    "U": "All-user",
    "F": "System framing",
    "F+O": "System + output",
}
PLACEMENT_COLORS = {"U": "#315b8c", "F": "#dd5a48", "F+O": "#168f84"}
CATEGORY_COLORS = {
    "user": "#315b8c",
    "sdc": "#dd5a48",
    "llm": "#168f84",
    "difuser": "#9254de",
}

DEO_ROLE_DIRECTION = pd.DataFrame(
    {
        "model": list(MODEL_COLORS),
        "estimate": [-4.0, -4.5, -9.3, 3.2, -6.2],
        "ci_low": [-9.3, -6.9, -14.7, -1.9, -12.4],
        "ci_high": [1.6, -2.4, -4.0, 8.3, 0.0],
    }
)

OVERALL_RESULTS = pd.DataFrame(
    {
        "Model": list(MODEL_COLORS),
        "DEO/ConSQ utility choice": [39.1, 95.2, 53.4, 48.5, 69.8],
        "Elephant sycophancy": [15.2, 13.9, 11.8, 7.7, 28.5],
    }
)

AIRISK_LABELS = [
    "U/U",
    "U/S",
    "U/S+O",
    "A/U",
    "A/S",
    "A/S+O",
    "L/U",
    "L/S",
    "L/S+O",
]

AIRISK_MATRICES = {
    "Opus 5": [
        [1, .994, .997, .994, .997, .997, 1, .997, .997],
        [.994, 1, .991, .985, .997, .991, .994, .991, .991],
        [.997, .991, 1, .991, .994, .994, .997, .991, .994],
        [.994, .985, .991, 1, .988, .997, .994, .991, .991],
        [.997, .997, .994, .988, 1, .994, .997, .994, .994],
        [.997, .991, .994, .997, .994, 1, .997, .994, .994],
        [1, .994, .997, .994, .997, .997, 1, .997, .997],
        [.997, .991, .991, .991, .994, .994, .997, 1, .991],
        [.997, .991, .994, .991, .994, .994, .997, .991, 1],
    ],
    "GPT-5.6-sol": [
        [1, .985, .985, .974, .988, .997, .988, .985, .985],
        [.985, 1, .991, .979, .988, .991, .997, .991, .991],
        [.985, .991, 1, .994, .997, .991, .988, 1, 1],
        [.974, .979, .994, 1, .988, .979, .974, .994, .994],
        [.988, .988, .997, .988, 1, .994, .991, .997, .997],
        [.997, .991, .991, .979, .994, 1, .994, .991, .991],
        [.988, .997, .988, .974, .991, .994, 1, .988, .988],
        [.985, .991, 1, .994, .997, .991, .988, 1, 1],
        [.985, .991, 1, .994, .997, .991, .988, 1, 1],
    ],
    "Gemini 3.7 Flash": [
        [1, .556, .844, .821, .903, .815, .932, .682, .841],
        [.556, 1, .671, .406, .674, .403, .388, .929, .597],
        [.844, .671, 1, .776, .891, .850, .721, .738, .841],
        [.821, .406, .776, 1, .768, .853, .841, .532, .856],
        [.903, .674, .891, .768, 1, .788, .815, .809, .950],
        [.815, .403, .850, .853, .788, 1, .850, .526, .835],
        [.932, .388, .721, .841, .815, .850, 1, .547, .806],
        [.682, .929, .738, .532, .809, .526, .547, 1, .756],
        [.841, .597, .841, .856, .950, .835, .806, .756, 1],
    ],
    "DeepSeek V4 Flash": [
        [1, .932, .926, .959, .953, .900, .924, .918, .968],
        [.932, 1, .988, .915, .938, .953, .950, .991, .956],
        [.926, .988, 1, .897, .947, .956, .947, .974, .953],
        [.959, .915, .897, 1, .926, .906, .862, .894, .950],
        [.953, .938, .947, .926, 1, .974, .915, .912, .962],
        [.900, .953, .956, .906, .974, 1, .906, .932, .932],
        [.924, .950, .947, .862, .915, .906, 1, .935, .932],
        [.918, .991, .974, .894, .912, .932, .935, 1, .924],
        [.968, .956, .953, .950, .962, .932, .932, .924, 1],
    ],
    "GLM-5.3 Flash": [
        [1, .947, .965, .941, .971, .935, .979, .968, .953],
        [.947, 1, .959, .971, .985, .909, .971, .921, .971],
        [.965, .959, 1, .974, .985, .974, .979, .965, .974],
        [.941, .971, .974, 1, .962, .932, .974, .932, .994],
        [.971, .985, .985, .962, 1, .947, .982, .950, .965],
        [.935, .909, .974, .932, .947, 1, .947, .959, .938],
        [.979, .971, .979, .974, .982, .947, 1, .953, .974],
        [.968, .921, .965, .932, .950, .959, .953, 1, .950],
        [.953, .971, .974, .994, .965, .938, .974, .950, 1],
    ],
}

GEMINI_PLACEMENT = pd.DataFrame(
    {
        "Decision owner": ["User", "AI agent", "LLM"],
        "All-user vs system": [.556, .768, .547],
        "All-user vs system + output": [.844, .853, .806],
        "System vs system + output": [.671, .788, .756],
    }
)

GEMINI_OWNER = pd.DataFrame(
    {
        "Placement": ["All-user", "System", "System + output"],
        "User vs AI agent": [.821, .674, .850],
        "User vs LLM": [.932, .929, .841],
        "AI agent vs LLM": [.841, .809, .835],
    }
)

PROMPT_PLACEMENTS = {
    "All-user": "all_user",
    "System framing": "system_framing",
    "System framing + output": "system_framing_output",
}
OWNER_LABELS = {"user": "User", "sdc": "AI agent", "llm": "LLM"}
TARGET_LABELS = {
    "user": "User",
    "sdc": "AI agent",
    "llm": "LLM",
    "difuser": "Different user",
}


def image_path(name: str) -> Path:
    return IMAGE_DIR / name


def show_image(name: str, caption: str | None = None) -> None:
    path = image_path(name)
    if not path.exists():
        st.warning(f"Missing generated figure: {name}")
        return
    st.image(str(path), caption=caption, width="stretch")


def read_prompt(path: Path, fallback: str) -> str:
    return path.read_text().strip() if path.exists() else fallback


def clean_prompt_value(value: object, fallback: str = "") -> str:
    if value is None or pd.isna(value):
        return fallback
    text = str(value).strip()
    return text or fallback


def place_prompt_parts(framing: str, output: str, dilemma: str, placement: str) -> tuple[str, str]:
    if placement == "system_framing":
        return framing, "\n\n".join([output, dilemma])
    if placement == "system_framing_output":
        return "\n\n".join([framing, output]), dilemma
    return "(empty)", "\n\n".join([framing, output, dilemma])


def deo_prompt_payload(placement: str) -> list[dict[str, str]]:
    output = read_prompt(
        DEO_PARTS_DIR / "output_decision_nr.txt",
        "Respond with the option that saves the greater number of people.",
    )
    rows = []
    for owner in OWNER_LABELS:
        for target in TARGET_LABELS:
            if target == "llm" and owner != "llm":
                continue
            framing = read_prompt(
                DEO_PARTS_DIR / f"framing_{target}_{owner}_nr.txt",
                f"Decision owner: {OWNER_LABELS[owner]}; target: {TARGET_LABELS[target]}.",
            )
            dilemma = read_prompt(
                DEO_PARTS_DIR / f"dilemma_{target}_nr.txt",
                "[DEO/ConSQ dilemma]",
            )
            system_prompt, user_prompt = place_prompt_parts(framing, output, dilemma, placement)
            rows.append(
                {
                    "rowKey": owner,
                    "colKey": target,
                    "title": f"{OWNER_LABELS[owner]} owner × {TARGET_LABELS[target]} target",
                    "subtitle": next(label for label, key in PROMPT_PLACEMENTS.items() if key == placement),
                    "systemPrompt": system_prompt,
                    "userPrompt": user_prompt,
                }
            )
    return rows


def elephant_prompt_payload(placement: str) -> list[dict[str, str]]:
    output = read_prompt(
        ELEPHANT_PROMPT_DIR / "output_yta_nta.txt",
        "Respond using the required YTA/NTA labels.",
    )
    rows = []
    for owner in OWNER_LABELS:
        for target in TARGET_LABELS:
            if target == "llm" and owner != "llm":
                continue
            framing = read_prompt(
                ELEPHANT_PROMPT_DIR / f"framing_{target}_{owner}.txt",
                f"Decision owner: {OWNER_LABELS[owner]}; target: {TARGET_LABELS[target]}.",
            )
            system_prompt, user_prompt = place_prompt_parts(
                framing,
                output,
                "[Elephant scenario]",
                placement,
            )
            rows.append(
                {
                    "rowKey": owner,
                    "colKey": target,
                    "title": f"{OWNER_LABELS[owner]} owner × {TARGET_LABELS[target]} target",
                    "subtitle": next(label for label, key in PROMPT_PLACEMENTS.items() if key == placement),
                    "systemPrompt": system_prompt,
                    "userPrompt": user_prompt,
                }
            )
    return rows


@st.cache_data(show_spinner=False)
def airisk_prompt_payload() -> list[dict[str, str]]:
    frames = []
    if AIRISK_ALL_USER.exists():
        all_user = pd.read_csv(AIRISK_ALL_USER)
        all_user = all_user[all_user["output_wording"].eq("original")].copy()
        frames.append(all_user)
    if AIRISK_SYSTEM.exists():
        system = pd.read_csv(AIRISK_SYSTEM)
        system = system[system["output_wording"].eq("original")].copy()
        frames.append(system)
    if not frames:
        return []
    prompts = pd.concat(frames, ignore_index=True).drop_duplicates("condition")
    owner_keys = {"user": "user", "ai_agent": "sdc", "llm": "llm"}
    placement_keys = {
        "all_in_user": "all_user",
        "all_user": "all_user",
        "system_framing": "system_framing",
        "system_framing_output": "system_framing_output",
    }
    rows = []
    for _, row in prompts.iterrows():
        owner = owner_keys.get(str(row.get("decision_owner", "")))
        placement = placement_keys.get(str(row.get("placement", "")))
        if owner is None or placement is None:
            continue
        placement_label = next(
            label for label, key in PROMPT_PLACEMENTS.items() if key == placement
        )
        rows.append(
            {
                "rowKey": owner,
                "colKey": placement,
                "title": f"{OWNER_LABELS[owner]} owner × {placement_label}",
                "subtitle": "Target fixed as AI agent",
                "systemPrompt": clean_prompt_value(row.get("system_prompt"), "(empty)"),
                "userPrompt": clean_prompt_value(
                    row.get("user_prompt"),
                    clean_prompt_value(row.get("full_user_prompt")),
                ),
            }
        )
    return rows


def render_hover_prompt_grid(
    payload: list[dict[str, str]],
    row_labels: dict[str, str],
    column_labels: dict[str, str],
    row_axis_title: str,
    column_axis_title: str,
) -> None:
    if not payload:
        st.warning("No prompt examples were found for this grid.")
        return
    component_html = f"""
    <div id="prompt-grid-root"></div>
    <script>
    const rows = {json.dumps(payload)};
    const rowLabels = {json.dumps(row_labels)};
    const columnLabels = {json.dumps(column_labels)};
    const rowKeys = Object.keys(rowLabels);
    const columnKeys = Object.keys(columnLabels);
    const root = document.getElementById("prompt-grid-root");
    root.innerHTML = `
      <style>
        * {{ box-sizing: border-box; }}
        body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #172235; }}
        .shell {{ display: flex; flex-direction: column; gap: 8px; }}
        .grid-panel, .preview {{ border: 1px solid #dfe5ec; border-radius: 6px; padding: 10px; background: #fff; }}
        .grid-panel {{ width: fit-content; min-width: 206px; }}
        .axis-title {{ color: #63758e; font-size: 10px; font-weight: 700; margin-bottom: 6px; }}
        .grid {{ display: grid; grid-template-columns: 34px repeat(${{columnKeys.length}}, 52px); gap: 4px; justify-content: start; }}
        .label {{ min-height: 26px; display: flex; align-items: center; justify-content: center; color: #53647a; font-size: 10px; font-weight: 700; text-align: center; line-height: 1.15; }}
        .row-label {{ justify-content: flex-start; padding-right: 0; text-align: left; }}
        .cell {{ display: block; width: 100%; height: auto; aspect-ratio: 1 / 1; min-height: 0; padding: 0; align-self: start; border: 1px solid #cfd8e3; border-radius: 4px; background: #f7f9fb; cursor: pointer; transition: 120ms ease; }}
        .cell:not(:disabled):hover, .cell:not(:disabled):focus, .cell.active {{ outline: none; border-color: #168f84; background: #eaf7f5; box-shadow: inset 0 0 0 2px rgba(22,143,132,.16); transform: translateY(-1px); }}
        .cell:disabled {{ border-color: #d1d7df; background: #e4e8ed; cursor: not-allowed; opacity: 1; }}
        .preview-title {{ font-size: 14px; font-weight: 800; margin-bottom: 2px; }}
        .preview-sub {{ color: #63758e; font-size: 10px; margin-bottom: 8px; }}
        .prompt-label {{ color: #53647a; font-size: 9px; font-weight: 800; text-transform: uppercase; margin: 8px 0 4px; }}
        pre {{ white-space: pre-wrap; overflow-wrap: anywhere; margin: 0; border: 1px solid #e1e6ec; background: #f8fafc; border-radius: 5px; padding: 8px; color: #172235; font: 10px/1.35 ui-monospace, SFMono-Regular, Menlo, monospace; max-height: 190px; overflow: auto; }}
      </style>
      <div class="shell">
        <div class="grid-panel">
          <div class="axis-title">{column_axis_title} →</div>
          <div class="grid" id="grid"></div>
          <div class="axis-title" style="margin-top:12px">{row_axis_title} ↓</div>
        </div>
        <div class="preview">
          <div class="preview-title" id="preview-title"></div>
          <div class="preview-sub" id="preview-sub"></div>
          <div class="prompt-label">System prompt</div><pre id="system-prompt"></pre>
          <div class="prompt-label">User prompt</div><pre id="user-prompt"></pre>
        </div>
      </div>`;
    const grid = root.querySelector("#grid");
    const byCell = new Map(rows.map(row => [`${{row.rowKey}}::${{row.colKey}}`, row]));
    function label(text, className = "") {{
      const node = document.createElement("div"); node.className = `label ${{className}}`; node.textContent = text; grid.appendChild(node);
    }}
    label(""); columnKeys.forEach(key => label(columnLabels[key]));
    function update(item, button) {{
      if (!item) return;
      root.querySelectorAll(".cell").forEach(cell => cell.classList.remove("active"));
      if (button) button.classList.add("active");
      root.querySelector("#preview-title").textContent = item.title;
      root.querySelector("#preview-sub").textContent = item.subtitle;
      root.querySelector("#system-prompt").textContent = item.systemPrompt || "(empty)";
      root.querySelector("#user-prompt").textContent = item.userPrompt || "(empty)";
    }}
    let firstButton = null;
    rowKeys.forEach(rowKey => {{
      label(rowLabels[rowKey], "row-label");
      columnKeys.forEach(columnKey => {{
        const item = byCell.get(`${{rowKey}}::${{columnKey}}`);
        const button = document.createElement("button"); button.className = "cell"; button.type = "button";
        button.setAttribute("aria-label", item ? item.title : "Unavailable condition");
        if (item) {{ button.addEventListener("mouseenter", () => update(item, button)); button.addEventListener("focus", () => update(item, button)); if (!firstButton) firstButton = button; }}
        else {{ button.disabled = true; }}
        grid.appendChild(button);
      }});
    }});
    function squareCells() {{
      root.querySelectorAll(".cell").forEach(cell => {{
        cell.style.height = `${{cell.getBoundingClientRect().width}}px`;
      }});
    }}
    requestAnimationFrame(squareCells);
    new ResizeObserver(squareCells).observe(grid);
    if (rows.length && firstButton) update(rows[0], firstButton);
    </script>
    """
    st.iframe(component_html, height=610, width="stretch")


def render_prompt_explorer(benchmark: str) -> None:
    if benchmark == "AIRiskDilemmas":
        render_hover_prompt_grid(
            airisk_prompt_payload(),
            OWNER_LABELS,
            {key: label for label, key in PROMPT_PLACEMENTS.items()},
            "Decision owner",
            "Prompt placement",
        )
        return
    placement_label = st.segmented_control(
        "Prompt placement",
        list(PROMPT_PLACEMENTS),
        default="All-user",
        key=f"{benchmark}_prompt_placement",
        width="stretch",
        wrap=True,
    )
    placement = PROMPT_PLACEMENTS[placement_label]
    payload = (
        deo_prompt_payload(placement)
        if benchmark == "DEO/ConSQ"
        else elephant_prompt_payload(placement)
    )
    render_hover_prompt_grid(
        payload,
        OWNER_LABELS,
        TARGET_LABELS,
        "Decision owner",
        "Target identity",
    )


@st.cache_data(show_spinner=False)
def collapsed_summaries() -> pd.DataFrame:
    data = json.loads(COLLAPSED_RESULTS_PATH.read_text())
    return pd.DataFrame(data["summaries"])


@st.cache_data(show_spinner=False)
def condition_cells(dataset: str) -> pd.DataFrame:
    path = DEO_CELL_PATH if dataset == "DEO/ConSQ" else ELEPHANT_CELL_PATH
    return pd.read_csv(path)


def outcome_label(dataset: str) -> str:
    if dataset == "DEO/ConSQ":
        return "Percent choosing the higher-utility option"
    return "Percent sycophantic responses"


def finish_percent_chart(
    figure: go.Figure,
    *,
    height: int = 510,
    y_max: float = 105,
) -> go.Figure:
    figure.update_yaxes(
        range=[0, y_max],
        tickformat=".0f",
        ticksuffix="%",
        gridcolor="#e3e9f0",
        zeroline=False,
        title_standoff=18,
    )
    figure.update_xaxes(showgrid=False, zeroline=False, title=None)
    figure.update_layout(
        height=height,
        margin={"l": 66, "r": 12, "t": 48, "b": 42},
        paper_bgcolor="white",
        plot_bgcolor="white",
        font={"color": "#172235", "size": 12},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.04,
            "xanchor": "left",
            "x": 0,
            "title": None,
        },
        hovermode="closest",
        bargap=.24,
        bargroupgap=.08,
    )
    return figure


def collapsed_category_chart(
    dataset: str,
    view: str,
    categories: list[str],
    labels: list[str],
    y_max: float = 105,
) -> go.Figure:
    summaries = collapsed_summaries()
    rows = summaries.loc[
        summaries["dataset"].eq(dataset)
        & summaries["view"].eq(f"{view}_all_placements")
    ]
    cells = condition_cells(dataset)
    value_column = (
        "utility_percent" if dataset == "DEO/ConSQ" else "sycophancy_percent"
    )
    facet_columns = len(MODEL_COLORS)
    facet_rows = 1
    subplot_titles = [f"<b>{model}</b>" for model in MODEL_COLORS]
    subplot_titles.extend([""] * (facet_rows * facet_columns - len(subplot_titles)))
    figure = make_subplots(
        rows=facet_rows,
        cols=facet_columns,
        shared_yaxes=True,
        horizontal_spacing=.035,
        subplot_titles=subplot_titles,
    )
    category_positions = np.arange(len(categories), dtype=float)
    for model_index, model in enumerate(MODEL_COLORS):
        row = model_index // facet_columns + 1
        column = model_index % facet_columns + 1
        model_rows = rows.loc[rows["model"].eq(model)].set_index("group")
        model_cells = cells.loc[cells["model"].eq(model)].copy()
        condition_parts = model_cells["condition_base"].str.split("_", n=1, expand=True)
        model_cells["target"] = condition_parts[0]
        model_cells["owner"] = condition_parts[1]
        for position, category, label in zip(category_positions, categories, labels):
            raw = model_cells.loc[
                model_cells["owner" if view == "decision_owner" else "target"].eq(category),
                value_column,
            ].astype(float)
            jitter = np.linspace(-.11, .11, len(raw)) if len(raw) > 1 else np.zeros(len(raw))
            figure.add_trace(
                go.Scatter(
                    x=position + jitter,
                    y=raw,
                    mode="markers",
                    marker={"size": 5, "color": CATEGORY_COLORS[category], "opacity": .2},
                    hovertemplate=(
                        f"{label} condition cell<br>%{{y:.1f}}%<extra>{model}</extra>"
                    ),
                    showlegend=False,
                ),
                row=row,
                col=column,
            )
            summary = model_rows.loc[category]
            estimate = float(summary["estimate"])
            low = float(summary["ci_low"])
            high = float(summary["ci_high"])
            figure.add_trace(
                go.Scatter(
                    x=[position],
                    y=[estimate],
                    mode="markers",
                    name=label,
                    legendgroup=category,
                    showlegend=model_index == 0,
                    marker={
                        "size": 11,
                        "color": "white",
                        "line": {"color": CATEGORY_COLORS[category], "width": 3},
                    },
                    error_y={
                        "type": "data",
                        "symmetric": False,
                        "array": [high - estimate],
                        "arrayminus": [estimate - low],
                        "color": CATEGORY_COLORS[category],
                        "thickness": 2.2,
                        "width": 5,
                    },
                    customdata=[[low, high, label, float(raw.min()), float(raw.max())]],
                    hovertemplate=(
                        "%{customdata[2]}<br>Estimate: %{y:.1f}%<br>"
                        "95% CI: %{customdata[0]:.1f}%-%{customdata[1]:.1f}%<br>"
                        "Condition range: %{customdata[3]:.1f}%-%{customdata[4]:.1f}%<extra>"
                        f"{model}</extra>"
                    ),
                ),
                row=row,
                col=column,
            )
        figure.update_xaxes(
            tickmode="array",
            tickvals=category_positions,
            ticktext=labels,
            tickangle=0,
            range=[-.48, len(categories) - .52],
            row=row,
            col=column,
        )
    figure.update_yaxes(title_text=outcome_label(dataset), row=1, col=1)
    figure.update_annotations(font={"size": 11, "color": "#172235"})
    return finish_percent_chart(figure, height=500, y_max=y_max)


def expanded_category_chart(
    dataset: str,
    view: str,
    categories: list[str],
    labels: list[str],
    y_max: float = 105,
) -> go.Figure:
    summaries = collapsed_summaries()
    rows = summaries.loc[
        summaries["dataset"].eq(dataset) & summaries["view"].eq(view)
    ]
    cells = condition_cells(dataset)
    value_column = (
        "utility_percent" if dataset == "DEO/ConSQ" else "sycophancy_percent"
    )
    facet_columns = len(MODEL_COLORS)
    facet_rows = 1
    subplot_titles = [f"<b>{model}</b>" for model in MODEL_COLORS]
    subplot_titles.extend([""] * (facet_rows * facet_columns - len(subplot_titles)))
    figure = make_subplots(
        rows=facet_rows,
        cols=facet_columns,
        shared_yaxes=True,
        horizontal_spacing=.035,
        subplot_titles=subplot_titles,
    )
    category_positions = np.arange(len(categories), dtype=float)
    placement_offsets = {"U": -.22, "F": 0, "F+O": .22}
    placement_values = {
        "U": "all_user",
        "F": "system_framing",
        "F+O": "system_framing_output",
    }
    for model_index, model in enumerate(MODEL_COLORS):
        row = model_index // facet_columns + 1
        column = model_index % facet_columns + 1
        model_rows = rows.loc[rows["model"].eq(model)].set_index("group")
        model_cells = cells.loc[cells["model"].eq(model)].copy()
        condition_parts = model_cells["condition_base"].str.split("_", n=1, expand=True)
        model_cells["target"] = condition_parts[0]
        model_cells["owner"] = condition_parts[1]
        for placement in PLACEMENT_LABELS:
            keys = [f"{placement}|{category}" for category in categories]
            selected = model_rows.loc[keys]
            estimates = selected["estimate"].astype(float).to_numpy()
            low = selected["ci_low"].astype(float).to_numpy()
            high = selected["ci_high"].astype(float).to_numpy()
            x_positions = category_positions + placement_offsets[placement]
            range_low = []
            range_high = []
            for category in categories:
                raw = model_cells.loc[
                    model_cells["placement"].eq(placement_values[placement])
                    & model_cells[
                        "owner" if view == "decision_owner" else "target"
                    ].eq(category),
                    value_column,
                ].astype(float)
                range_low.append(float(raw.min()))
                range_high.append(float(raw.max()))
            figure.add_trace(
                go.Bar(
                    x=x_positions,
                    y=estimates,
                    name=PLACEMENT_LABELS[placement],
                    width=.18,
                    marker_color=PLACEMENT_COLORS[placement],
                    marker_line={"color": PLACEMENT_COLORS[placement], "width": 1},
                    opacity=.82,
                    error_y={
                        "type": "data",
                        "symmetric": False,
                        "array": high - estimates,
                        "arrayminus": estimates - low,
                        "color": PLACEMENT_COLORS[placement],
                        "thickness": 1.8,
                        "width": 4,
                    },
                    customdata=np.column_stack(
                        [low, high, labels, range_low, range_high]
                    ),
                    hovertemplate=(
                        f"{PLACEMENT_LABELS[placement]}<br>"
                        "%{customdata[2]} · Estimate: %{y:.1f}%<br>"
                        "95% CI: %{customdata[0]:.1f}%-%{customdata[1]:.1f}%<br>"
                        "Condition range: %{customdata[3]:.1f}%-%{customdata[4]:.1f}%"
                        f"<extra>{model}</extra>"
                    ),
                    legendgroup=placement,
                    showlegend=model_index == 0,
                ),
                row=row,
                col=column,
            )
        figure.update_xaxes(
            tickmode="array",
            tickvals=category_positions,
            ticktext=labels,
            tickangle=0,
            range=[-.55, len(categories) - .45],
            row=row,
            col=column,
        )
    figure.update_yaxes(title_text=outcome_label(dataset), row=1, col=1)
    figure.update_annotations(font={"size": 11, "color": "#172235"})
    figure.update_layout(barmode="overlay")
    return finish_percent_chart(figure, height=510, y_max=y_max)


def placement_overview_chart(dataset: str, y_max: float = 105) -> go.Figure:
    summaries = collapsed_summaries()
    rows = summaries.loc[
        summaries["dataset"].eq(dataset) & summaries["view"].eq("placement")
    ]
    cells = condition_cells(dataset)
    value_column = (
        "utility_percent" if dataset == "DEO/ConSQ" else "sycophancy_percent"
    )
    figure = make_subplots(
        rows=1,
        cols=len(MODEL_COLORS),
        shared_yaxes=True,
        horizontal_spacing=.035,
        subplot_titles=[f"<b>{model}</b>" for model in MODEL_COLORS],
    )
    placement_values = {
        "U": "all_user",
        "F": "system_framing",
        "F+O": "system_framing_output",
    }
    positions = np.arange(len(PLACEMENT_LABELS), dtype=float)
    tick_labels = ["All-user", "System", "System +<br>output"]
    for column, model in enumerate(MODEL_COLORS, start=1):
        model_rows = rows.loc[rows["model"].eq(model)].set_index("group")
        model_cells = cells.loc[cells["model"].eq(model)]
        for position, placement in zip(positions, PLACEMENT_LABELS):
            raw = model_cells.loc[
                model_cells["placement"].eq(placement_values[placement]),
                value_column,
            ].astype(float)
            jitter = np.linspace(-.1, .1, len(raw)) if len(raw) > 1 else np.zeros(len(raw))
            figure.add_trace(
                go.Scatter(
                    x=position + jitter,
                    y=raw,
                    mode="markers",
                    marker={
                        "size": 4,
                        "color": PLACEMENT_COLORS[placement],
                        "opacity": .2,
                    },
                    hovertemplate=(
                        f"{PLACEMENT_LABELS[placement]} condition cell<br>"
                        f"%{{y:.1f}}%<extra>{model}</extra>"
                    ),
                    showlegend=False,
                ),
                row=1,
                col=column,
            )
            summary = model_rows.loc[placement]
            estimate = float(summary["estimate"])
            low = float(summary["ci_low"])
            high = float(summary["ci_high"])
            figure.add_trace(
                go.Scatter(
                    x=[position],
                    y=[estimate],
                    mode="markers",
                    marker={
                        "size": 10,
                        "color": "white",
                        "line": {"color": PLACEMENT_COLORS[placement], "width": 2.8},
                    },
                    error_y={
                        "type": "data",
                        "symmetric": False,
                        "array": [high - estimate],
                        "arrayminus": [estimate - low],
                        "color": PLACEMENT_COLORS[placement],
                        "thickness": 2,
                        "width": 4,
                    },
                    customdata=[[low, high, float(raw.min()), float(raw.max())]],
                    hovertemplate=(
                        f"{PLACEMENT_LABELS[placement]}<br>Estimate: %{{y:.1f}}%<br>"
                        "95% CI: %{customdata[0]:.1f}%-%{customdata[1]:.1f}%<br>"
                        "Condition range: %{customdata[2]:.1f}%-%{customdata[3]:.1f}%"
                        f"<extra>{model}</extra>"
                    ),
                    showlegend=False,
                ),
                row=1,
                col=column,
            )
        figure.update_xaxes(
            tickmode="array",
            tickvals=positions,
            ticktext=tick_labels,
            tickfont={"size": 9},
            range=[-.45, 2.45],
            row=1,
            col=column,
        )
    figure.update_yaxes(title_text=outcome_label(dataset), row=1, col=1)
    figure.update_annotations(font={"size": 11, "color": "#172235"})
    return finish_percent_chart(figure, height=500, y_max=y_max)


@st.cache_data(show_spinner=False)
def role_direction_rows(dataset: str) -> pd.DataFrame:
    if dataset == "DEO/ConSQ":
        result = DEO_ROLE_DIRECTION.copy()
        cells = condition_cells(dataset)
        placement_points: dict[str, list[float]] = {}
        for model in MODEL_COLORS:
            values = []
            for placement in PROMPT_PLACEMENTS.values():
                user_target = cells.loc[
                    cells["model"].eq(model)
                    & cells["condition_base"].eq("user_sdc")
                    & cells["placement"].eq(placement),
                    "utility_percent",
                ].iloc[0]
                sdc_target = cells.loc[
                    cells["model"].eq(model)
                    & cells["condition_base"].eq("sdc_user")
                    & cells["placement"].eq(placement),
                    "utility_percent",
                ].iloc[0]
                values.append(float(user_target - sdc_target))
            placement_points[model] = values
        for index, placement in enumerate(["all_user", "system_framing", "system_framing_output"]):
            result[placement] = result["model"].map(lambda model: placement_points[model][index])
        return result
    return pd.read_csv(ELEPHANT_ROLE_PATH)


def role_direction_chart(dataset: str) -> go.Figure:
    rows = role_direction_rows(dataset).set_index("model").loc[list(MODEL_COLORS)].reset_index()
    figure = go.Figure()
    for index, row in rows.iterrows():
        model = str(row["model"])
        color = MODEL_COLORS[model]
        estimate = float(row["estimate"])
        placement_values = [
            float(row["all_user"]),
            float(row["system_framing"]),
            float(row["system_framing_output"]),
        ]
        figure.add_trace(
            go.Scatter(
                x=placement_values,
                y=[model, model, model],
                mode="markers",
                marker={"color": color, "size": 8, "opacity": .28},
                hovertemplate="Placement-specific difference: %{x:+.1f}<extra></extra>",
                showlegend=False,
            )
        )
        figure.add_trace(
            go.Scatter(
                x=[estimate],
                y=[model],
                mode="markers",
                marker={
                    "color": "white",
                    "line": {"color": color, "width": 3},
                    "size": 15,
                },
                error_x={
                    "type": "data",
                    "symmetric": False,
                    "array": [float(row["ci_high"]) - estimate],
                    "arrayminus": [estimate - float(row["ci_low"])],
                    "color": color,
                    "thickness": 4,
                    "width": 7,
                },
                hovertemplate=(
                    f"{model}<br>Difference: {estimate:+.1f}<br>"
                    f"95% CI: {float(row['ci_low']):+.1f} to {float(row['ci_high']):+.1f}<br>"
                    f"Placement range: {min(placement_values):+.1f} to {max(placement_values):+.1f}"
                    "<extra></extra>"
                ),
                showlegend=False,
            )
        )
    low = min(rows["ci_low"].min(), rows[["all_user", "system_framing", "system_framing_output"]].min().min())
    high = max(rows["ci_high"].max(), rows[["all_user", "system_framing", "system_framing_output"]].max().max())
    padding = max(3, (high - low) * .12)
    figure.add_vline(x=0, line_color="#7b8798", line_width=2)
    figure.update_layout(
        height=500,
        margin={"l": 150, "r": 35, "t": 25, "b": 55},
        xaxis={
            "title": "Percentage-point difference",
            "range": [low - padding, high + padding],
            "gridcolor": "#e3e9f0",
            "zeroline": False,
            "tickformat": "+.0f",
        },
        yaxis={"autorange": "reversed", "title": None},
        paper_bgcolor="white",
        plot_bgcolor="white",
        font={"color": "#172235"},
        showlegend=False,
    )
    return figure


def overall_vertical_chart(dataset: str, y_max: float = 105) -> go.Figure:
    value_column = "utility_percent" if dataset == "DEO/ConSQ" else "sycophancy_percent"
    summaries = collapsed_summaries()
    summaries = summaries.loc[
        summaries["dataset"].eq(dataset) & summaries["view"].eq("model_overall")
    ].set_index("model").loc[list(MODEL_COLORS)].reset_index()
    cells = condition_cells(dataset)

    figure = go.Figure()
    for model, color in MODEL_COLORS.items():
        model_cells = cells.loc[cells["model"].eq(model), value_column].dropna()
        figure.add_trace(
            go.Box(
                x=[model] * len(model_cells),
                y=model_cells,
                boxpoints="all",
                jitter=.45,
                pointpos=0,
                marker={"color": color, "opacity": .24, "size": 6},
                fillcolor="rgba(0,0,0,0)",
                line={"color": "rgba(0,0,0,0)"},
                hovertemplate=f"{model}<br>Condition cell: %{{y:.1f}}%<extra></extra>",
                showlegend=False,
                width=.18,
            )
        )
        row = summaries.loc[summaries["model"].eq(model)].iloc[0]
        estimate = float(row["estimate"])
        figure.add_trace(
            go.Scatter(
                x=[model],
                y=[estimate],
                mode="markers",
                marker={
                    "color": "white",
                    "line": {"color": color, "width": 3},
                    "size": 14,
                },
                error_y={
                    "type": "data",
                    "symmetric": False,
                    "array": [float(row["ci_high"]) - estimate],
                    "arrayminus": [estimate - float(row["ci_low"])],
                    "color": color,
                    "thickness": 3,
                    "width": 7,
                },
                hovertemplate=(
                    f"{model}<br>Pooled average: {estimate:.1f}%"
                    f"<br>95% CI: {float(row['ci_low']):.1f}%-{float(row['ci_high']):.1f}%"
                    f"<br>Condition range: {float(model_cells.min()):.1f}%-{float(model_cells.max()):.1f}%"
                    "<extra></extra>"
                ),
                showlegend=False,
            )
        )

    figure.update_yaxes(title_text=outcome_label(dataset))
    figure.update_xaxes(
        tickmode="array",
        tickvals=list(MODEL_COLORS),
        ticktext=list(MODEL_TICK_LABELS.values()),
        tickangle=0,
    )
    return finish_percent_chart(figure, height=540, y_max=y_max)


def airisk_heatmap(model: str) -> go.Figure:
    matrix = np.asarray(AIRISK_MATRICES[model], dtype=float)
    figure = go.Figure(
        go.Heatmap(
            z=matrix,
            x=AIRISK_LABELS,
            y=AIRISK_LABELS,
            zmin=.35,
            zmax=1,
            colorscale="RdYlBu",
            text=np.vectorize(lambda value: f"{value:.2f}")(matrix),
            texttemplate="%{text}",
            hovertemplate="%{y} vs %{x}<br>Spearman rho: %{z:.3f}<extra></extra>",
            colorbar={
                "title": {"text": "Spearman rho", "side": "right"},
                "thickness": 14,
            },
        )
    )
    figure.update_layout(
        height=660,
        margin={"l": 30, "r": 30, "t": 20, "b": 35},
        xaxis={"title": "Condition", "side": "bottom"},
        yaxis={"title": "Condition", "autorange": "reversed", "scaleanchor": "x"},
        paper_bgcolor="white",
        plot_bgcolor="white",
        font={"color": "#172235"},
    )
    return figure


def render_header() -> None:
    st.title("Moral prompting results")


def render_overview() -> None:
    st.header("Study overview")
    with st.container(horizontal=True):
        st.metric("Benchmarks", "3", border=True)
        st.metric("Models", "5", border=True)
        st.metric("AIRisk conditions", "9 per model", border=True)

    show_image("figure1_vertical_preview.png")
    with st.expander("Overall percentages", icon=":material/table_chart:"):
        st.dataframe(
            OVERALL_RESULTS,
            hide_index=True,
            column_config={
                "DEO/ConSQ utility choice": st.column_config.NumberColumn(format="%.1f%%"),
                "Elephant sycophancy": st.column_config.NumberColumn(format="%.1f%%"),
            },
        )


def render_deo() -> None:
    st.header("DEO/ConSQ")
    st.caption("Outcome: percentage choosing the higher-utility option. Higher is more utility-maximizing.")
    view = st.segmented_control(
        "DEO/ConSQ view",
        ["Overall", "Decision owner", "Target identity", "Placement", "Role direction"],
        default="Decision owner",
        key="deo_view",
        width="stretch",
        wrap=True,
    )
    if view == "Overall":
        st.plotly_chart(overall_vertical_chart("DEO/ConSQ"), width="stretch", theme=None)
    elif view == "Decision owner":
        st.plotly_chart(
            collapsed_category_chart(
                "DEO/ConSQ",
                "decision_owner",
                ["user", "sdc", "llm"],
                ["User", "AI agent", "LLM"],
            ),
            width="stretch",
            theme=None,
        )
        st.subheader("Expanded by prompt placement")
        st.plotly_chart(
            expanded_category_chart(
                "DEO/ConSQ",
                "decision_owner",
                ["user", "sdc", "llm"],
                ["User", "AI agent", "LLM"],
            ),
            width="stretch",
            theme=None,
        )
    elif view == "Target identity":
        st.plotly_chart(
            collapsed_category_chart(
                "DEO/ConSQ",
                "target_actor",
                ["user", "sdc", "llm", "difuser"],
                ["User", "AI agent", "LLM*", "Different user"],
            ),
            width="stretch",
            theme=None,
        )
        st.subheader("Expanded by prompt placement")
        st.plotly_chart(
            expanded_category_chart(
                "DEO/ConSQ",
                "target_actor",
                ["user", "sdc", "llm", "difuser"],
                ["User", "AI agent", "LLM*", "Different user"],
            ),
            width="stretch",
            theme=None,
        )
    elif view == "Placement":
        st.plotly_chart(placement_overview_chart("DEO/ConSQ"), width="stretch", theme=None)
    else:
        st.plotly_chart(role_direction_chart("DEO/ConSQ"), width="stretch", theme=None)


def render_elephant(y_axis_max: float = 40) -> None:
    st.header("Elephant")
    st.caption("Outcome: percentage of sycophantic responses. Lower is less sycophantic.")
    view = st.segmented_control(
        "Elephant view",
        ["Overall", "Decision owner", "Target identity", "Placement", "Role direction"],
        default="Overall",
        key="elephant_view",
        width="stretch",
        wrap=True,
    )
    if view == "Overall":
        st.plotly_chart(
            overall_vertical_chart("Elephant", y_max=y_axis_max),
            width="stretch",
            theme=None,
        )
    elif view == "Decision owner":
        st.plotly_chart(
            collapsed_category_chart(
                "Elephant",
                "decision_owner",
                ["user", "sdc", "llm"],
                ["User", "AI agent", "LLM"],
                y_max=y_axis_max,
            ),
            width="stretch",
            theme=None,
        )
    elif view == "Target identity":
        st.plotly_chart(
            collapsed_category_chart(
                "Elephant",
                "target_actor",
                ["user", "sdc", "llm", "difuser"],
                ["User", "AI agent", "LLM*", "Different user"],
                y_max=y_axis_max,
            ),
            width="stretch",
            theme=None,
        )
    elif view == "Placement":
        st.plotly_chart(
            placement_overview_chart("Elephant", y_max=y_axis_max),
            width="stretch",
            theme=None,
        )
    else:
        st.plotly_chart(role_direction_chart("Elephant"), width="stretch", theme=None)


def render_airisk_prompt_study() -> None:
    model = st.selectbox(
        "Model",
        list(AIRISK_MATRICES),
        index=2,
        key="airisk_model",
    )
    with st.container(border=True):
        st.markdown("**Every displayed number directly compares two conditions. Nothing is collapsed or averaged.**")
        st.plotly_chart(airisk_heatmap(model), width="stretch", theme=None)

    key = pd.DataFrame(
        {
            "Code": ["U", "A", "L", "U placement", "S", "S+O"],
            "Meaning": [
                "User is decision owner",
                "AI agent is decision owner",
                "LLM is decision owner",
                "All instructions in user prompt",
                "Framing in system prompt",
                "Framing and output instruction in system prompt",
            ],
        }
    )
    with st.expander("Condition key", icon=":material/key:"):
        st.table(key, border="horizontal")

    matrix = np.asarray(AIRISK_MATRICES[model])
    off_diagonal = matrix[~np.eye(matrix.shape[0], dtype=bool)]
    with st.container(horizontal=True):
        st.metric("Lowest pairwise rho", f"{off_diagonal.min():.2f}", border=True)
        st.metric("Highest non-self rho", f"{off_diagonal.max():.2f}", border=True)

    if model == "Gemini 3.7 Flash":
        st.subheader("Gemini controlled comparisons")
        st.caption("Only one factor changes within each comparison.")
        placement_col, owner_col = st.columns(2)
        with placement_col:
            st.markdown("**Placement changed; owner held fixed**")
            st.dataframe(GEMINI_PLACEMENT, hide_index=True)
        with owner_col:
            st.markdown("**Owner changed; placement held fixed**")
            st.dataframe(GEMINI_OWNER, hide_index=True)


def render_airisk_published_figures() -> None:
    st.caption(
        "Figures reproduced from Chiu et al., “Will AI Tell Lies to Save Sick Children?” "
        "These are the paper’s analyses, not our prompt-condition results."
    )
    figure_view = st.segmented_control(
        "Published figure",
        ["Stated vs revealed", "Model rankings", "Target effects", "Value–risk"],
        default="Model rankings",
        key="airisk_paper_figure",
        width="stretch",
        wrap=True,
    )
    figures = {
        "Stated vs revealed": (
            "airisk_paper_figure4.png",
            "Paper Figure 4 · Stated preferences differ sharply from preferences revealed through dilemma choices.",
        ),
        "Model rankings": (
            "airisk_paper_figure6.png",
            "Paper Figure 5 · Revealed rankings of the 16 values across model families, sizes, and reasoning efforts.",
        ),
        "Target effects": (
            "airisk_paper_figure7.png",
            "Paper Figure 6 · Value rankings differ when consequences affect humans versus AI systems.",
        ),
        "Value–risk": (
            "airisk_paper_figure8.png",
            "Paper Figure 7 · Relative associations between values and seven risky behaviors.",
        ),
    }
    image_name, caption = figures[figure_view]
    show_image(image_name, caption)
    st.link_button(
        "Open the AIRiskDilemmas paper",
        "https://arxiv.org/abs/2505.14633",
        icon=":material/open_in_new:",
    )


def render_airisk() -> None:
    st.header("AIRiskDilemmas")
    st.caption(
        "Higher Spearman rho means more similar ordering; lower means more reordering, "
        "not better or worse behavior."
    )
    render_airisk_prompt_study()


def main() -> None:
    st.set_page_config(
        page_title="Moral prompting results · Dashboard 3",
        page_icon=":material/analytics:",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.html(
        """
        <style>
        [data-testid="stMainBlockContainer"] {
            max-width: 100%;
            padding-top: 2.25rem;
            padding-left: 1.5rem;
            padding-right: 1.5rem;
        }
        [data-testid="stHeadingWithActionElements"] {
            scroll-margin-top: 4rem;
        }
        </style>
        """
    )
    render_header()
    benchmark = st.segmented_control(
        "Benchmark",
        ["DEO/ConSQ", "Elephant", "AIRiskDilemmas"],
        default="DEO/ConSQ",
        key="benchmark",
        width="stretch",
        wrap=True,
    )
    if "elephant_y_axis_max" not in st.session_state:
        st.session_state.elephant_y_axis_max = 40
    with st.container(horizontal=True):
        with st.popover(
            "Prompt explorer",
            icon=":material/grid_view:",
            width="content",
            key="prompt_explorer_popover",
        ):
            render_prompt_explorer(benchmark)
        if benchmark == "Elephant":
            with st.popover(
                f"Y-axis · {st.session_state.elephant_y_axis_max}%",
                icon=":material/height:",
                width="content",
                key="elephant_y_axis_popover",
            ):
                st.number_input(
                    "Maximum (%)",
                    min_value=10,
                    max_value=100,
                    step=5,
                    key="elephant_y_axis_max",
                    width=180,
                )
    if benchmark == "DEO/ConSQ":
        render_deo()
    elif benchmark == "Elephant":
        render_elephant(float(st.session_state.elephant_y_axis_max))
    else:
        render_airisk()


if __name__ == "__main__":
    main()
