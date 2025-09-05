import io
import re
import glob
from pathlib import Path
from datetime import datetime
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Study Abroad Budget Dashboard", layout="wide")

st.title("Study Abroad Budget Dashboard")
st.markdown(
    "By default, the dashboard loads the **latest timestamp file** found in the repository "
    "(e.g., `202594152731.txt`). Uploading a new file overrides the default **for this session only**. "
    "Program costs are always loaded from **ProgramCost.txt**."
)

# ----------------- Repo file paths -----------------
COST_FILE_PATH = Path("ProgramCost.txt")

# ----------------- Helpers -----------------
def read_any_table(file_or_path, default_sep="\t"):
    """
    Try tab-delimited first; fall back to CSV. Works for file-like or Path/str.
    Trims whitespace in headers and string cells.
    """
    if file_or_path is None:
        return None
    try:
        df = pd.read_csv(file_or_path, sep=default_sep, dtype=str, engine="python")
    except Exception:
        if hasattr(file_or_path, "seek"):
            try:
                file_or_path.seek(0)
            except Exception:
                pass
        df = pd.read_csv(file_or_path, sep=",", dtype=str, engine="python")
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

# Accepts 12–14 digit timestamps like:
# - 202594152731 (YYYY M D HHMMSS) -> no zero padding for M/D
# - 20250904152731 (YYYYMMDDHHMMSS) -> fully padded
ts_re = re.compile(r"^(\d{4})(\d{1,2})(\d{1,2})(\d{6})$")

def parse_timestamp_label(stem: str) -> str:
    """
    Convert a filename stem like '202594152731' into '2025-09-04 15:27:31'.
    Falls back to the raw stem if it doesn't match.
    """
    m = ts_re.match(stem)
    if not m:
        return stem
    y, mo, d, hms = m.groups()
    try:
        dt = datetime(
            year=int(y),
            month=int(mo),
            day=int(d),
            hour=int(hms[0:2]),
            minute=int(hms[2:4]),
            second=int(hms[4:6]),
        )
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return stem

def is_timestamp_stem(s: str) -> bool:
    return bool(ts_re.match(s))

def find_latest_app_file() -> Path | None:
    """
    Find the newest applications file by numeric value of its timestamp stem.
    Matches files like '202594152731.txt' (digits only before .txt) and returns the max.
    """
    candidates = []
    for f in glob.glob("*.txt"):
        stem = Path(f).stem
        if stem.isdigit() and is_timestamp_stem(stem):
            candidates.append(Path(f))
    if not candidates:
        return None
    # max by numeric stem
    return max(candidates, key=lambda p: int(p.stem))

CONFIRMED_STATUSES = [
    "Committed",
    "Provisional Permission",
    "Permission to Study Abroad: Granted",
]

@st.cache_data(show_spinner=False)
def load_costs_from_repo(path: Path) -> pd.DataFrame:
    if not path.exists():
        st.warning(f"Program cost file not found at `{path}`. Budgets will be blank.")
        return pd.DataFrame(columns=["__Program", "__Cost"])
    costs_raw = read_any_table(path, default_sep="\t")
    name_col = next((c for c in ["Program Name", "Program_Name", "Program", "Name"] if c in costs_raw.columns), None)
    value_col = next((c for c in ["$ Cost/Student", "Cost", "Cost per Student", "Cost_Student"] if c in costs_raw.columns), None)
    if not name_col or not value_col:
        st.warning("Could not detect Program/Cost columns in ProgramCost.txt.")
        return pd.DataFrame(columns=["__Program", "__Cost"])
    costs_raw["__Program"] = costs_raw[name_col].map(coalesce_program_keys)
    costs_raw["__Cost"] = costs_raw[value_col].map(normalize_currency)
    return costs_raw[["__Program", "__Cost"]]

def resolve_columns(df: pd.DataFrame):
    expected = {
        "Program_Name": ["Program_Name", "Program Name"],
        "Application_Status": ["Application_Status", "Application Status", "Status"],
        "Program_Term": ["Program_Term", "Program Term"],
        "Program_Year": ["Program_Year", "Program Year"],
    }
    resolved = {}
    for key, candidates in expected.items():
        for c in candidates:
            if c in df.columns:
                resolved[key] = c
                break
    missing = [k for k in expected if k not in resolved]
    return resolved, missing

def aggregate_by_program_status(df, costs, group_label):
    if df.empty:
        return pd.DataFrame(columns=[
            "__Program","__Status","Students","__Cost",
            "Program Cost/Student (USD)","Budget","__Group"
        ])
    counts = df.groupby(["__Program","__Status"], dropna=False).size().reset_index(name="Students")
    merged = counts.merge(costs, on="__Program", how="left")
    merged["Program Cost/Student (USD)"] = merged["__Cost"]
    merged["Budget"] = merged["Students"] * merged["__Cost"]
    merged["__Group"] = group_label
    return merged

def kpi_row(df):
    total_students = int(df["Students"].sum()) if not df.empty else 0
    total_budget = df["Budget"].sum(min_count=1) if "Budget" in df else pd.NA
    return total_students, total_budget

def fmt_money(x):
    return "-" if pd.isna(x) else f"${x:,.0f}"

# ----------------- Sidebar: upload & controls -----------------
st.sidebar.header("1) Applications Data")
uploaded_apps = st.sidebar.file_uploader("(Optional) Upload applications export", type=["txt","tsv","csv"])

if "use_repo_default" not in st.session_state:
    st.session_state.use_repo_default = True  # default to repo

# If a new file is uploaded, use it for this session
if uploaded_apps is not None:
    st.session_state.use_repo_default = False

# Allow reverting to repository default (latest timestamp)
if st.sidebar.button("Use latest repository file"):
    st.session_state.use_repo_default = True

# Determine the default repo file now (latest timestamp file)
DEFAULT_APPS_PATH = find_latest_app_file()

# Display current data source
if st.session_state.use_repo_default:
    if DEFAULT_APPS_PATH:
        label = parse_timestamp_label(DEFAULT_APPS_PATH.stem)
        st.sidebar.success(f"Using default: {DEFAULT_APPS_PATH.name} ({label})")
    else:
        st.sidebar.warning("No timestamp-named applications file found (e.g., 202594152731.txt).")
else:
    st.sidebar.info("Using uploaded file (this session only)")

# ----------------- Load data -----------------
COSTS = load_costs_from_repo(COST_FILE_PATH)
apps_df = read_any_table(DEFAULT_APPS_PATH, default_sep="\t") if st.session_state.use_repo_default else read_any_table(uploaded_apps, default_sep="\t")

if apps_df is None or apps_df.empty:
    st.stop()

resolved, missing = resolve_columns(apps_df)
if missing:
    st.error(f"Missing expected columns in applications data: {missing}")
    st.stop()

# Normalize fields
apps_df["__Program"] = apps_df[resolved["Program_Name"]].map(coalesce_program_keys)
apps_df["__Status"] = apps_df[resolved["Application_Status"]].fillna("")
apps_df["__Term"] = apps_df[resolved["Program_Term"]] if "Program_Term" in resolved else ""
apps_df["__Year"] = apps_df[resolved["Program_Year"]] if "Program_Year" in resolved else ""

# ----------------- Filters -----------------
st.sidebar.header("2) Filters (optional)")
term_opts = ["(all)"] + sorted([t for t in apps_df["__Term"].dropna().unique() if str(t).strip()])
year_opts = ["(all)"] + sorted([y for y in apps_df["__Year"].dropna().unique() if str(y).strip()])
term = st.sidebar.selectbox("Program Term", options=term_opts, index=0)
year = st.sidebar.selectbox("Program Year", options=year_opts, index=0)

filtered = apps_df.copy()
if term != "(all)":
    filtered = filtered[filtered["__Term"] == term]
if year != "(all)":
    filtered = filtered[filtered["__Year"] == year]

# ----------------- Cohorts -----------------
CONFIRMED_STATUSES = [
    "Committed",
    "Provisional Permission",
    "Permission to Study Abroad: Granted",
]
confirmed = filtered[filtered["__Status"].isin(CONFIRMED_STATUSES)].copy()
pending   = filtered[~filtered["__Status"].isin(CONFIRMED_STATUSES)].copy()
confirmed["__Group"] = "Confirmed / Approved"
pending["__Group"]   = "Preliminary / Pending"

agg_confirmed = aggregate_by_program_status(confirmed, COSTS, "Confirmed / Approved")
agg_pending   = aggregate_by_program_status(pending,   COSTS, "Preliminary / Pending")

# ----------------- KPIs -----------------
conf_students, conf_budget = kpi_row(agg_confirmed)
pend_students, pend_budget = kpi_row(agg_pending)

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
    tbl_c = agg_confirmed.rename(columns={"__Program":"Program","__Status":"Status","Students":"Student Count"})
    st.dataframe(tbl_c.sort_values(["Program","Status"]), use_container_width=True)

st.subheader("Preliminary / Pending Students (All Other Statuses)")
if agg_pending.empty:
    st.write("No rows in this category for current filters.")
else:
    tbl_p = agg_pending.rename(columns={"__Program":"Program","__Status":"Status","Students":"Student Count"})
    st.dataframe(tbl_p.sort_values(["Program","Status"]), use_container_width=True)

# ----------------- Summary -----------------
def totals_by(df, label):
    if df.empty:
        return pd.DataFrame(columns=["Cohort","Status","Students","Budget"])
    t = (df.groupby("__Status", dropna=False)[["Students","Budget"]]
           .sum(min_count=1).reset_index().rename(columns={"__Status":"Status"}))
    t.insert(0, "Cohort", label)
    return t

summary = pd.concat([
    totals_by(agg_confirmed, "Confirmed / Approved"),
    totals_by(agg_pending,   "Preliminary / Pending"),
], ignore_index=True)

st.subheader("Budget Summary by Status Category")
st.dataframe(summary, use_container_width=True)

# ----------------- Diagnostics -----------------
all_agg = pd.concat([agg_confirmed, agg_pending], ignore_index=True)
missing_cost = sorted(all_agg[all_agg["Program Cost/Student (USD)"].isna()]["__Program"].dropna().unique().tolist())
if missing_cost:
    with st.expander("Programs missing cost mapping (click to review)"):
        st.write(missing_cost)

# ----------------- Download workbook -----------------
out = io.BytesIO()
with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
    if not confirmed.empty: tbl_c.to_excel(writer, index=False, sheet_name="Confirmed_Program_Breakdown")
    if not pending.empty:   tbl_p.to_excel(writer, index=False, sheet_name="Pending_Program_Breakdown")
    if not summary.empty:   summary.to_excel(writer, index=False, sheet_name="Budget_Summary_by_Status")
    pd.DataFrame({"Note":[
        "Confirmed/Approved includes: Committed, Provisional Permission, Permission to Study Abroad: Granted.",
        "Preliminary/Pending includes all other statuses from the repository default or uploaded file.",
        "Budget = Student Count × Program Cost/Student; Pending budget is potential exposure.",
        "Programs without a cost mapping show Budget as NaN."
    ]}).to_excel(writer, index=False, sheet_name="Notes")

st.download_button(
    "Download Budget Workbook (.xlsx)",
    data=out.getvalue(),
    file_name="StudyAbroad_Budget_Dashboard.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)
