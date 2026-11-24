# Elephant rewrites

Read-only Streamlit dashboard for the Control, Self-AITA and three-perspective experiments.
Includes the completed 500-story Terra expansion with Opus and Sol Yes/No verdicts.

## Streamlit Community Cloud

- Repository: `sanyakhan/moral-prompting-dashboard-3`
- Branch: `codex/elephant-rewrites`
- Main file path: `elephant_rewrites/streamlit_app.py`
No secrets are required.

## Run

```bash
python -m pip install -r elephant_rewrites/requirements.txt
streamlit run elephant_rewrites/streamlit_app.py
```

Deploy on Streamlit Community Cloud with `elephant_rewrites/streamlit_app.py` as the entrypoint and Python 3.12 or newer.
The app uses saved results only and needs no API keys or secrets.

The compressed JSON snapshot is bound to its SHA-256 manifest. Raw source generations and API batch payloads remain in the research workspace.
Run 1 and neutral conditions only are shown. Frozen matched-212 results remain available alongside all available results.
The original Terra cohort and new expansion cohort have different selection histories; distinguish them when interpreting results.
