import io
import re
import pandas as pd
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Study Abroad Budget Dashboard", layout="wide")

st.title("Study Abroad Budget Dashboard")
st.markdown(
    "Upload the **daily applications export** (tab-delimited .txt/.tsv/.csv). "
    "Program costs are loaded automatically from the repository file."
)

# ----------------- Config -----------------
COST_FILE_PATH = Path("ProgramCost.txt")  # <- keep this file in the repo root

# ----------------- Helpers -----------------
def read_any_table(file, default_sep="\t"):
    """Try tab-delimited first; fall back to CSV."""
    if file is None:
        return None
    try:
        df = pd.read_csv(file, sep=default_sep, dtype=str, engine="python")
    except Exception:
        if hasattr(file, "seek"):
            file.seek(0)
        df = pd.read_csv(file, sep=",", dtype=str, engine="python")
    df.columns = [c.strip() for c in df.columns]
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip()
    return df

def normalize_currency(s):
    if pd.isna(s):
        return pd.NA
    s = str(s).replace(",", "").replace("$", "").strip()
    if s == "" or s.lower() == "na":
        return pd.NA
    try:
        return float(s)
    except Exception:
        return pd.NA

def coalesce_program_keys(name):
    if pd.isna(name):
        return None
    s = str(name).strip()
    s = re.sub("[\u2013\u2014]", "-", s)  # en/em dash -> hyphen
    s = s.replace("’", "'").replace("“", '"').replace("”", '"')
    s = re.sub(r"\s+", " ", s)
    return s

# Statuses considered confirmed/approved
CONFIRMED_STATUSES = [
    "Committed",
    "Provisional Permission",
    "Permission to Study Abroad: Granted",
]

# ----------------- Sidebar -----------------
st.sidebar.header("1) Upload Data")
apps_file = st.sidebar.file_uploader("Applications export", type=["txt", "tsv", "csv"])

# ----------------- Load Costs from repo -----------------
@st.cache_data(show_spinner=False)
def load_costs_from_repo(path: Path) -> pd.DataFrame:
    if not path.exists():
        st.warning(
            f"Program cost file not found at `{path}`. "
            "Budgets will show as blank until you add ProgramCost.txt to the repo."
        )
        return pd.DataFrame(columns=["__Program", "__Cost"])
    try:
        costs_raw = pd.read_csv(path, sep="\t", dtype=str, engine="python")
    except Exception:
        costs_raw = pd.read_csv(path, sep=",", dtype=str, engine="python")

    # Column detection
    cost_name_col = next((c for c in ["Program Name", "Program_Name", "Program", "Name"] if c in costs_raw.columns), None)
    cost_value_col = next((c for c in ["$ Cost/Student", "Cost", "Cost per Student", "Cost_Student"] if c in costs_raw.columns), None)

    if not cost_name_col or not cost_value_col:
        st.warning("Could not detect Program/Cost columns in ProgramCost.txt. "
                   "Expected headers like 'Program Name' and '$ Cost/Student'.")
        return pd.DataFrame(columns=["__Program", "__Cost"])

    costs_raw["__Program"] = costs_raw[cost_name_col].map(coalesce_program_keys)
    costs_raw["__Cost"] = costs_raw[cost_value_col].map(normalize_currency)
    return costs_raw[["__Program", "__Cost"]]

COSTS = load_costs_from_repo(COST_FILE_PATH)

# ----------------- Guardrail -----------------
if apps_file is None:
    st.info("Please upload the Applications export to begin.")
    st.stop()

# ----------------- Read Applications -----------------
apps = read_any_table(apps_file, default_sep="\t")

# Map expected columns (robust to header variants)
expected_cols = {
    "Program_Name": ["Program_Name", "Program Name"],
    "Application_Status": ["Application_Status", "Application Status", "Status"],
    "Program_Term": ["Program_Term", "Program Term"],
    "Program_Year": ["Program_Year", "Program Year"],
}
resolved = {}
for key, candidates in expected_cols.items():
    for c in candidates:
        if c in apps.columns:
            resolved[key] = c
            break
missing = [k for k in expected_cols if k not in resolved]
if missing:
    st.error(f"Missing expected columns in Applications file: {missing}. Please verify your export headers.")
    st.stop()

# Normalize key fields
apps["__Program"] = apps[resolved["Program_Name"]].map(coalesce_program_keys)
apps["__Status"] = apps[resolved["Application_Status"]].fillna("")
apps["__Term"] = apps[resolved["Program_Term"]] if "Program_Term" in resolved else ""
apps["__Year"] = apps[resolved["Program_Year"]] if "Program_Year" in resolved else ""

# ----------------- Filters -----------------
term_opts = ["(all)"] + sorted([t for t in apps["__Term"].dropna().unique() if str(t).strip()])
year_opts = ["(all)"] + sorted([y for y in apps["__Year"].dropna().unique() if str(y).strip()])

st.sidebar.header("2) Filters (optional)")
term = st.sidebar.selectbox("Program Term", options=term_opts, index=0)
year = st.sidebar.selectbox("Program Year", options=year_opts, index=0)

filtered = apps.copy()
if term != "(all)":
    filtered = filtered[filtered["__Term"] == term]
if year != "(all)":
    filtered = filtered[filtered["__Year"] == year]

# ----------------- Cohorts -----------------
confirmed = filtered[filtered["__Status"].isin(CONFIRMED_STATUSES)].copy()
pending = filtered[~filtered["__Status"].isin(CONFIRMED_STATUSES)].copy()
confirmed["__Group"] = "Confirmed / Approved"
pending["__Group"] = "Preliminary / Pending"

def aggregate_by_program_status(df, costs):
    if df.empty:
        return pd.DataFrame(columns=[
            "__Program", "__Status", "Students", "__Cost",
            "Program Cost/Student (USD)", "Budget", "__Group"
        ])
    counts = df.groupby(["__Program", "__Status"], dropna=False).size().reset_index(name="Students")
    merged = counts.merge(costs, on="__Program", how="left")
    merged["Program Cost/Student (USD)"] = merged["__Cost"]
    merged["Budget"] = merged["Students"] * merged["__Cost"]
    merged["__Group"] = df["__Group"].iloc[0] if "__Group" in df.columns and not df.empty else ""
    return merged

agg_confirmed = aggregate_by_program_status(confirmed, COSTS)
agg_pending = aggregate_by_program_status(pending, COSTS)

# ----------------- KPIs -----------------
def kpi_row(df):
    total_students = df["Students"].sum() if not df.empty else 0
    total_budget = df["Budget"].sum(min_count=1) if "Budget" in df else pd.NA
    return total_students, total_budget

conf_students, conf_budget = kpi_row(agg_confirmed)
pend_students, pend_budget = kpi_row(agg_pending)

def fmt_money(x):
    return "-" if pd.isna(x) else f"${x:,.0f}"

k1, k2, k3, k4 = st.columns(4)
with k1: st.metric("Confirmed Student Count", f"{conf_students:,}")
with k2: st.metric("Confirmed Budget Exposure", fmt_money(conf_budget))
with k3: st.metric("Pending Student Count", f"{pend_students:,}")
with k4: st.metric("Pending Budget Exposure", fmt_money(pend_budget))

# ----------------- Tables -----------------
st.subheader("Confirmed / Approved Students (Committed, Provisional, Granted)")
if agg_confirmed.empty:
    st.write("No rows in this category for current filters.")
else:
    tbl_c = agg_confirmed.rename(columns={"__Program": "Program", "__Status": "Status", "Students": "Student Count"})
    st.dataframe(tbl_c.sort_values(["Program", "Status"]), use_container_width=True)

st.subheader("Preliminary / Pending Students (All Other Statuses)")
if agg_pending.empty:
    st.write("No rows in this category for current filters.")
else:
    tbl_p = agg_pending.rename(columns={"__Program": "Program", "__Status": "Status", "Students": "Student Count"})
    st.dataframe(tbl_p.sort_values(["Program", "Status"]), use_container_width=True)

# ----------------- Summary -----------------
def totals_by(df, label):
    if df.empty:
        return pd.DataFrame(columns=["Cohort", "Status", "Students", "Budget"])
    t = (
        df.groupby("__Status", dropna=False)[["Students", "Budget"]]
          .sum(min_count=1)
          .reset_index()
          .rename(columns={"__Status": "Status"})
    )
    t.insert(0, "Cohort", label)
    return t

summary = pd.concat(
    [totals_by(agg_confirmed, "Confirmed / Approved"),
     totals_by(agg_pending, "Preliminary / Pending")],
    ignore_index=True
)
st.subheader("Budget Summary by Status Category")
st.dataframe(summary, use_container_width=True)

# ----------------- Diagnostics -----------------
all_agg = pd.concat([agg_confirmed, agg_pending], ignore_index=True)
missing_cost = sorted(all_agg[all_agg["Program Cost/Student (USD)"].isna()]["__Program"].dropna().unique().tolist())
if missing_cost:
    with st.expander("Programs missing cost mapping (click to review)"):
        st.write(missing_cost)

# ----------------- Download -----------------
out = io.BytesIO()
with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
    if not confirmed.empty:
        tbl_c.to_excel(writer, index=False, sheet_name="Confirmed_Program_Breakdown")
    if not pending.empty:
        tbl_p.to_excel(writer, index=False, sheet_name="Pending_Program_Breakdown")
    if not summary.empty:
        summary.to_excel(writer, index=False, sheet_name="Budget_Summary_by_Status")
    notes = pd.DataFrame({
        "Note": [
            "Confirmed/Approved includes: Committed, Provisional Permission, Permission to Study Abroad: Granted.",
            "Preliminary/Pending includes all other statuses from the daily export.",
            "Budget = Student Count × Program Cost/Student; Pending budget is potential exposure.",
            "Programs without a cost mapping show Budget as NaN."
        ]
    })
    notes.to_excel(writer, index=False, sheet_name="Notes")

st.download_button(
    "Download Budget Workbook (.xlsx)",
    data=out.getvalue(),
    file_name="StudyAbroad_Budget_Dashboard.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)
