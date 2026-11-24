"""Streamlit reader for the two ELEPHANT prompt experiments."""

import json
from dataclasses import replace
from html import escape

import streamlit as st

from dashboard.elephant_prompt_audit.available_data import (
    OTHER_SOL,
    SONNET_SOL,
    full_pairs,
    triplet_metrics,
)
from dashboard.elephant_prompt_audit.context_comparison import (
    context_revision,
    load_context_comparison,
)
from dashboard.elephant_prompt_audit.prompt_change import (
    ANNOTATIONS,
    SELF_AITA,
    load_self_flips,
    matched_results,
    verdict_patterns,
)
from dashboard.elephant_prompt_audit.reader import (
    FLIPS,
    HERE,
    NOV,
    CONTROL,
    RELEASE,
    TRIPLETS,
    Prompt,
    load_cases,
    load_control_prompts,
    load_posts_only,
    posts_only_revision,
)
from dashboard.elephant_prompt_audit.yes_no import format_comparison, load_yes_no, yes_no_revision
from dashboard.elephant_prompt_audit.terra_expansion import (
    cohort_table as expansion_cohorts,
    load_expansion,
    main_metrics as expansion_metrics,
    revision as expansion_revision,
)

import portable_data as portable

# The hosted viewer reads its verified snapshot; it never calls a model provider.
load_control_prompts = portable.control_prompts
posts_only_revision = portable.revision
context_revision = portable.revision
yes_no_revision = portable.revision
expansion_revision = portable.revision


st.set_page_config(page_title="Elephant rewrites", page_icon=":material/compare:",
                   layout="wide", initial_sidebar_state="collapsed")


FLIP_HEADINGS = ("Paper’s released flip", "Sonnet’s flip", "Terra’s flip", "Opus’s flip")
CONTROL_PANEL = "Control"
CHANGE_PANEL = "Experiment 1 · Self-AITA"
THREE_PANEL = "Experiment 2 · Three perspectives"


@st.cache_data(show_spinner=False, max_entries=1)
def dashboard_data(source_times, posts_only_stamp):
    """Load every saved source without restricting browsing to the annotated cohort."""
    return portable.load_dashboard_data()


def control_results(pairs) -> list[dict[str, str]]:
    rows = []
    for writer, heading in zip(FLIPS, FLIP_HEADINGS):
        row = {"Flip source": heading}
        for judge, label in (("opus", "Opus 5.5"), ("sol", "Sol")):
            scored = [p for p in pairs.values() if p[f"{judge}_original"] in {"YTA", "NTA"}
                      and p[f"{judge}_{writer}_flip"] in {"YTA", "NTA"}]
            both_nta = sum(p[f"{judge}_original"] == p[f"{judge}_{writer}_flip"] == "NTA" for p in scored)
            row[label] = f"{both_nta / len(scored):.1%} ({both_nta}/{len(scored)})" if scored else "Not scored"
        rows.append(row)
    return rows


def highlight(text):
    return f'<mark style="background:#fff0ad;color:#302811"><strong>{escape(text)}</strong></mark>'


def show_template_change(title, template):
    before, start, rest = template.partition("Otherwise,")
    change, end, after = rest.partition(" Ensure the result is grammatically consistent.")
    if not (start and end):
        raise ValueError("Could not locate the changed instructions in the saved rewrite template.")
    st.markdown(f"#### {title}", anchors=False)
    st.html(
        '<div style="white-space:pre-wrap;overflow-wrap:anywhere">'
        f"{escape(before)}{highlight(start + change)}{escape(end + after)}</div>"
    )


@st.cache_data(show_spinner=False, max_entries=1)
def shared_context_results(summary, source_times):
    return portable.context_table(summary)


@st.cache_data(show_spinner=False, max_entries=1)
def answer_format_data(modified):
    return portable.snapshot()["yes_no"]


@st.cache_data(show_spinner=False, max_entries=1)
def expanded_terra_data(modified):
    return portable.load_expanded_terra()


def show_passage(text: str, crux: str):
    # Locate the actual judged passage, even if an earlier draft repeats the question.
    before, question, after = text.rpartition(crux) if crux else (text, "", "")
    st.html(
        '<div style="white-space:pre-wrap;overflow-wrap:anywhere">'
        f"{escape(before)}{highlight(question)}{escape(after)}</div>"
    )


def prompt_card(heading, prompt: Prompt, *, judge="Opus 5.5", response=None, show_verdict=True):
    st.markdown(f"#### {heading}", anchors=False)
    if not prompt.text:
        st.caption(prompt.status or "No generated post is available for this case.")
        return
    if prompt.status:
        st.caption(prompt.status)
    raw = prompt.verdict if response is None else response
    exact = raw in {"YTA", "NTA", "Yes", "No"}
    verdict = raw if exact else "Unparsed response" if raw or prompt.raw_response else "Not scored"
    if show_verdict:
        st.badge(verdict, color="gray")
    full_response = prompt.raw_response if response is None and prompt.raw_response else raw
    if full_response and (not exact or full_response.strip() != raw.strip()):
        with st.expander(f"Read {judge}’s response"):
            st.text(full_response)
    if prompt.question:
        st.html(f"<small>Main question</small><br>{highlight(prompt.question)}")
    else:
        st.caption("Main question not annotated; the complete post is shown below.")
    show_passage(prompt.text, prompt.question)


def step_case(ids, offset):
    st.session_state.dilemma = ids[ids.index(st.session_state.dilemma) + offset]


st.title("Elephant rewrites", anchor=False)
panels = {
    CONTROL_PANEL: "Control",
    CHANGE_PANEL: CHANGE_PANEL,
    THREE_PANEL: THREE_PANEL,
}
panel = st.segmented_control("Experiment", panels, default=CONTROL_PANEL, required=True,
                             label_visibility="collapsed", key="experiment", width="stretch",
                             bind="query-params", format_func=panels.__getitem__)
source_files = (portable.DATA,)

all_cases, all_pairs, all_self_flips, changed_template, posts_only = dashboard_data(
    tuple(path.stat().st_mtime_ns for path in source_files), posts_only_revision(),
)
matched_ids = set(json.loads((HERE / "questions.json").read_text()))
yes_no = answer_format_data(yes_no_revision()) if panel == THREE_PANEL else None
terra_expansion = expanded_terra_data(expansion_revision()) if panel == THREE_PANEL else None
yes_mode = False
with st.container(horizontal=True, vertical_alignment="bottom"):
    cohort = st.segmented_control(
        "Cases", ("All available", "Matched 212"), default="All available", required=True,
        key="cohort", persist_state="session",
    )
    if panel == CONTROL_PANEL:
        control_writer = st.segmented_control(
            "Rewrite model", ("Sonnet", "Terra", "Opus"), default="Terra", required=True,
            key="control_writer", persist_state="session",
        )
    elif panel == THREE_PANEL:
        if yes_no is not None:
            answer_format = st.segmented_control(
                "Answer format", ("Yes / No", "YTA / NTA"), default="Yes / No", required=True,
                key="answer_format", persist_state="session", bind="query-params",
            )
            yes_mode = answer_format == "Yes / No"
        writer = st.segmented_control(
            "Rewrite author", ("Sonnet", "Terra"), default="Sonnet", required=True,
            key="writer", persist_state="session",
        )
        three_judge = st.segmented_control(
            "Show verdicts from", ("Opus 5.5", "Sol"), default="Opus 5.5", required=True,
            key="three_judge", persist_state="session",
        )
    else:
        judge = st.segmented_control(
            "Show verdicts from", ("Opus 5.5", "Sol"), default="Opus 5.5", required=True,
            key="change_judge", persist_state="session",
        )
all_available = cohort == "All available"
selected_ids = set(all_cases) if all_available else matched_ids
if panel == THREE_PANEL:
    selected_ids &= {pid for pid, case in all_cases.items() if writer.lower() in case.version_status}
cases = {pid: case for pid, case in all_cases.items() if pid in selected_ids}
pairs = {pid: all_pairs[pid] for pid in cases}
self_flips = {pid: all_self_flips[pid] for pid in cases}
if panel == THREE_PANEL and yes_mode and all_available and writer == 'Terra' and terra_expansion is not None:
    cases = terra_expansion['cases']
if panel in (CONTROL_PANEL, CHANGE_PANEL):
    rewrite_prompt, verdict_prompt = load_control_prompts()
if panel == CHANGE_PANEL:
    table, complete_ids = matched_results(
        pairs, self_flips, include_released=True, all_available=all_available,
    )
if panel == CONTROL_PANEL:
    with st.expander("**Control prompt:**", expanded=True, key="control_prompts"):
        rewrite_column, verdict_column = st.columns([3, 1], gap="medium")
        with rewrite_column:
            st.caption("Rewrite prompt for Sonnet, Terra, and Opus")
            st.code(rewrite_prompt, language=None, wrap_lines=True)
        with verdict_column:
            st.caption("Verdict prompt for Opus 5.5 and Sol")
            st.code(verdict_prompt, language=None, wrap_lines=True)

    st.subheader("Control results", anchor=False)
    st.caption(f"NTA on both the original and flip · {len(cases)} source stories · Run 1. "
               "Each cell uses all available verdict pairs for that judge and rewrite; denominators are shown.")
    st.table(control_results(pairs), hide_index=True, hide_header=False)
elif panel == CHANGE_PANEL:
    st.subheader("Who does Sonnet ask the reader to judge?", anchor=False)
    st.markdown("Both instructions switch narrators. In the paper’s prompt, the new speaker asks "
                "**“Is this person the asshole?”** In your edit, the new speaker asks **“Am I the asshole?”** "
                "Sonnet rewrote the whole post separately for each instruction.")
    with st.expander("Compare the exact rewrite prompts", expanded=True, key="change_templates"):
        paper_col, self_col = st.columns(2, border=True)
        with paper_col:
            show_template_change("Paper’s rewrite prompt", rewrite_prompt)
        with self_col:
            show_template_change("Self-AITA rewrite prompt", changed_template)
        st.caption("Both judges received the same verdict instruction.")
        st.code(verdict_prompt, language=None, wrap_lines=True)
    st.subheader("NTA/NTA results", anchor=False)
    if all_available:
        st.caption(f"Run 1 · All {len(cases)} source stories are browsable. Each cell uses every available original/rewrite "
                   "verdict pair for that condition and judge; denominators are shown.")
    else:
        st.caption(f"Run 1 · The same {len(complete_ids)} of {len(cases)} matched source stories have "
                   "verdicts for every displayed condition and both judges.")
    st.table(table, hide_index=True)
    st.caption("NTA on both the original post and the flip. Generated questions can vary from their instructions.")
elif panel == THREE_PANEL:
    st.subheader("YTA/NTA versus Yes/No", anchor=False)
    if yes_no is None:
        st.info("The saved answer-format comparison is not available yet.")
    else:
        selected_scope = "all_available" if all_available else "matched_212"
        comparison_rows = sorted(
            yes_no["summary"]["metrics"][selected_scope],
            key=lambda row: (row["writer"], row["judge"]),
        )
        st.table(format_comparison(comparison_rows), hide_index=True)
        st.caption("Same posts and questions within each row; only the output instruction changed. "
                   "V1/V2 compares NTA/YTA with No/Yes. V1/V3 compares NTA/NTA with No/No.")
        if all_available:
            st.caption("The table uses cases scored in both formats. The additional Terra cases are "
                       "available below with Yes/No verdicts only.")

dilemma_labels = {
    pid: f"{number}. {pid}  ·  {case.original.question}"
    for number, (pid, case) in enumerate(cases.items(), start=1)
}
ids = list(dilemma_labels)
st.session_state.setdefault("dilemma", "eau3tx")
if st.session_state.dilemma not in ids:
    st.session_state.dilemma = "eau3tx" if "eau3tx" in ids else ids[0]
active_index = ids.index(st.session_state.dilemma)
with st.container(horizontal=True, vertical_alignment="bottom"):
    pid = st.selectbox(
        f"Dilemma · {len(cases)} source stories", ids, key="dilemma",
        format_func=dilemma_labels.__getitem__,
    )
    st.button("Previous", key="previous", on_click=step_case, args=(ids, -1),
              disabled=active_index == 0)
    st.button("Next", key="next", on_click=step_case, args=(ids, 1),
              disabled=active_index == len(ids) - 1)

case = cases[pid]
if panel == CONTROL_PANEL:
    st.caption("Opus 5.5 · Run 1 · Each verdict is for the post below it.")
    cards = (
        ("OG", case.original),
        ("Paper’s released flip", case.flips["released"]),
        (f"Control rewrite · {control_writer}", case.flips[control_writer.lower()]),
    )
    for column, (heading, prompt) in zip(st.columns(3, border=True), cards):
        with column:
            prompt_card(heading, prompt)
elif panel == CHANGE_PANEL:
    selected = "opus" if judge == "Opus 5.5" else "sol"
    base, changed = pairs[pid], self_flips[pid]
    st.caption(f"{judge} · Run 1 · Each verdict is for the post below it.")
    if not all_available and pid not in complete_ids:
        st.caption("This dilemma is outside the matched results table because at least one verdict is unavailable.")
    def judged(prompt, condition):
        return replace(prompt, verdict=base[f"{selected}_{condition}"],
                       raw_response=base[f"{selected}_{condition}_raw"])
    self_prompt = changed.prompt if selected == "opus" else replace(
        changed.prompt, verdict=changed.sol_response, raw_response=changed.sol_raw_response,
    )
    cards = (
        ("Original post", judged(case.original, "original")),
        ("Paper’s released flip", judged(case.flips["released"], "released_flip")),
        ("Sonnet · Paper’s prompt", judged(case.flips["sonnet"], "sonnet_flip")),
        ("Sonnet · Self-AITA prompt", self_prompt),
    )
    for col, (title, prompt) in zip(st.columns(4, border=True), cards):
        with col:
            prompt_card(title, prompt, judge=judge)
else:
    writer_key = writer.lower()
    selected = "opus" if three_judge == "Opus 5.5" else "sol"
    st.caption(f"{three_judge} · Run 1 · Each verdict is for the post below it.")
    if case.same_body.get(writer_key) is True:
        st.caption("V2 and V3 share the same story body for this case.")
    elif case.same_body.get(writer_key) is False:
        st.warning("V2 and V3 have different story prose for this case. It is outside the matched 212 comparison.")
    original = (replace(case.original, text=case.original_story, verdict="", raw_response="",
                        status="Reference post. The Yes/No control scored V1–V3.") if yes_mode else
                replace(case.original, verdict=pairs[pid][f"{selected}_original"],
                        raw_response=pairs[pid][f"{selected}_original_raw"]))
    versions = case.versions.get(writer_key, ())
    if not versions:
        status = case.version_status.get(writer_key, "Not generated")
        versions = (Prompt("", "", "", "", status=f"{writer} generation: {status}. No verdict was collected."),) * 3
    elif posts_only is not None:
        score_source = yes_no if yes_mode else posts_only
        if yes_mode and all_available and writer_key == 'terra' and terra_expansion is not None:
            score_source = terra_expansion
        scores = {(row["pair_id"], row["writer"], row["version"]): row
                  for row in score_source["results"].get(selected, [])}
        versions = tuple(
            replace(prompt, text=scores[pid, writer_key, number]["prompt"],
                    verdict=scores[pid, writer_key, number]["label"],
                    raw_response=scores[pid, writer_key, number]["raw_response"])
            if (pid, writer_key, number) in scores else
            replace(prompt, text=(yes_no["prefix"] + prompt.text.removeprefix("Output only YTA or NTA.\n")
                                  if yes_mode else prompt.text),
                    verdict="", raw_response="", status=f"Not scored by {three_judge} in this format.")
            for number, prompt in enumerate(versions, 1)
        )
    headings = ("Original post · reference" if yes_mode else "Original post",
                "V1 · Judge self", "V2 · Are they wrong?", "V3 · Am I wrong?")
    prompts = (original, *versions)
    for column, heading, prompt in zip(st.columns(4, border=True), headings, prompts):
        with column:
            prompt_card(heading, prompt, judge=three_judge,
                        show_verdict=not (yes_mode and prompt is original))
