import io
import re
import glob
import hashlib
from pathlib import Path
from datetime import datetime
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Study Abroad Budget Dashboard", layout="wide")

st.title("Study Abroad Budget Dashboard")
st.markdown(
    "• Defaults to the latest timestamp-named applications file in the repository (e.g., `202594152731.txt`).\n"
    "• Uploading a file overrides the default **for this session**.\n"
    "• Program costs are loaded from **ProgramCost.txt** (auto-refresh on changes).\n"
    "• Budgeting counts **unique students** (one final application per student): "
    "**Committed** > **Granted** > **Provisional**; ties break to **highest program cost**."
)

# ----------------- Repo file paths -----------------
COST_FILE_PATH = Path("ProgramCost.txt")

# ----------------- Helpers -----------------
def read_any_table(file_or_path, default_sep="\t"):
    """
    Read tab-delimited first; fall back to CSV.
    Works for file-like objects and Path/str. Trims whitespace.
    """
    if file_or_path is None:
        return None
    try:
        df = pd.read_csv(file_or_path, sep=default_sep, dtype=str, engine="python")
    except Exception:
        # reset pointer if file-like
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

# Accepts 12–14 digit timestamps like 202594152731 (YYYY M D HHMMSS) or fully padded 20250904152731
ts_re = re.compile(r"^(\d{4})(\d{1,2})(\d{1,2})(\d{6})$")

def parse_timestamp_label(stem: str) -> str:
    """Convert '202594152731' -> '2025-09-04 15:27:31'; fallback to stem if not parseable."""
    m = ts_re.match(stem)
    if not m:
        return stem
    y, mo, d, hms = m.groups()
    try:
        dt = datetime(int(y), int(mo), int(d), int(hms[0:2]), int(hms[2:4]), int(hms[4:6]))
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return stem

def is_timestamp_stem(s: str) -> bool:
    return bool(ts_re.match(s))

def find_latest_app_file() -> Path | None:
    """Pick the newest applications file by numeric value of its timestamp stem (e.g., 202594152731.txt)."""
    candidates = []
    for f in glob.glob("*.txt"):
        stem = Path(f).stem
        if stem.isdigit() and is_timestamp_stem(stem):
            candidates.append(Path(f))
    if not candidates:
        return None
    return max(candidates, key=lambda p: int(p.stem))

# Status priority (higher is better)
STATUS_PRIORITY = {
    "Committed": 3,
    "Permission to Study Abroad: Granted": 2,
    "Provisional Permission": 1
    # others -> 0
}
def status_rank(s): return STATUS_PRIORITY.get(s, 0)

def build_student_key(df: pd.DataFrame) -> pd.Series:
    """Choose the most reliable identifier available; fallback to name/email combo."""
    candidates = [
        "UR ID", "UR_ID", "Student_ID", "Student Id", "ID",
        "Email", "Email_Address", "Email Address", "Student Email",
        "NetID", "Net Id"
    ]
    found = next((c for c in candidates if c in df.columns), None)
    if found:
        return df[found].astype(str).str.strip().str.lower().replace({"nan": ""})
    # Fallback: First/Last name combo
    fn = df.columns[df.columns.str.lower().str.contains("first")].tolist()
    ln = df.columns[df.columns.str.lower().str.contains("last")].tolist()
    if fn and ln:
        return (df[fn[0]].astype(str).str.strip().str.lower() + "||" +
                df[ln[0]].astype(str).str.strip().str.lower())
    # Final fallback: email + program
    return (df.get("Email", pd.Series([""]*len(df))).astype(str).str.lower() + "||" +
            df.get("__Program", pd.Series([""]*len(df))).astype(str).str.lower())

def dedup_students_by_priority_then_cost(df_apps: pd.DataFrame, df_costs: pd.DataFrame) -> pd.DataFrame:
    """
    Keep exactly ONE row per student:
      1) Highest status priority wins (Committed > Granted > Provisional > others)
      2) If tie, choose the row with the highest program cost
      3) If still tied, keep the first occurrence (stable)
    """
    if df_apps.empty:
        return df_apps

    # Map program -> cost
    costs = df_costs.rename(columns={"__Program": "__Program_cost_key", "__Cost": "__CostValue"}).copy()

    tmp = df_apps.copy()
    tmp["__StudentKey"] = build_student_key(tmp)
    tmp["__Rank"] = tmp["__Status"].map(status_rank)

    # Join cost using normalized program name
    tmp = tmp.merge(
        costs[["__Program_cost_key", "__CostValue"]],
        left_on="__Program", right_on="__Program_cost_key", how="left"
    )

    # Treat missing cost as 0 for comparison
    tmp["__CostValue"] = tmp["__CostValue"].fillna(0.0)

    # Sort: higher rank first, higher cost first; stable to preserve original order for ties
    tmp = tmp.sort_values(["__Rank", "__CostValue"], ascending=[False, False], kind="stable")

    # Drop duplicates per student, keeping best-ranked, highest-cost row
    tmp = tmp.drop_duplicates(subset="__StudentKey", keep="first")

    # Clean up temp columns
    tmp = tmp.drop(columns=["__StudentKey", "__Rank", "__Program_cost_key", "__CostValue"], errors="ignore")
    return tmp

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

def file_md5(path: Path) -> str:
    """Hash a file's contents so cache invalidates when it changes."""
    if not path.exists():
        return ""
    return hashlib.md5(path.read_bytes()).hexdigest()

# ----------------- Sidebar: actions & upload -----------------
# Manual refresh option for admins/colleagues
if st.sidebar.button("Refresh data from repository"):
    st.cache_data.clear()
    st.experimental_rerun()

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

# Determine default repo file (latest timestamp)
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

# ----------------- Load costs with cache keyed on file hash -----------------
@st.cache_data(show_spinner=False, ttl=0)
def load_costs_from_repo(path: Path, content_hash: str) -> pd.DataFrame:
    """
    Cached by content_hash so any change to ProgramCost.txt auto-invalidates.
    ttl=0 => no time-based expiry; only hash or manual refresh changes.
    """
    if not path.exists():
        st.warning(f"Program cost file not found at `{path}`. Budgets will be blank.")
        return pd.DataFrame(columns=["__Program", "__Cost"])
    costs_raw = read_any_table(path, default_sep="\t")
    name_col  = next((c for c in ["Program Name", "Program_Name", "Program", "Name"] if c in costs_raw.columns), None)
    value_col = next((c for c in ["$ Cost/Student", "Cost", "Cost per Student", "Cost_Student"] if c in costs_raw.columns), None)
    if not name_col or not value_col:
        st.warning("Could not detect Program/Cost columns in ProgramCost.txt.")
        return pd.DataFrame(columns=["__Program", "__Cost"])
    costs_raw["__Program"] = costs_raw[name_col].map(coalesce_program_keys)
    costs_raw["__Cost"]    = costs_raw[value_col].map(normalize_currency)
    return costs_raw[["__Program", "__Cost"]]

COSTS = load_costs_from_repo(COST_FILE_PATH, file_md5(COST_FILE_PATH))

# ----------------- Load applications -----------------
apps_df = read_any_table(DEFAULT_APPS_PATH, default_sep="\t") if st.session_state.use_repo_default else read_any_table(uploaded_apps, default_sep="\t")
if apps_df is None or apps_df.empty:
    st.stop()

resolved, missing = resolve_columns(apps_df)
if missing:
    st.error(f"Missing expected columns in applications data: {missing}")
    st.stop()

# Normalize fields
apps_df["__Program"] = apps_df[resolved["Program_Name"]].map(coalesce_program_keys)
apps_df["__Status"]  = apps_df[resolved["Application_Status"]].fillna("")
apps_df["__Term"]    = apps_df[resolved["Program_Term"]] if "Program_Term" in resolved else ""
apps_df["__Year"]    = apps_df[resolved["Program_Year"]] if "Program_Year" in resolved else ""

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

# ----------------- UNIQUE-STUDENT DEDUP -----------------
# One row per student: highest status priority, then highest cost
filtered = dedup_students_by_priority_then_cost(filtered, COSTS)

# ----------------- Cohorts (after dedup) -----------------
CONFIRMED_STATUSES = ["Committed", "Permission to Study Abroad: Granted", "Provisional Permission"]
confirmed = filtered[filtered["__Status"].isin(CONFIRMED_STATUSES)].copy()
pending   = filtered[~filtered["__Status"].isin(CONFIRMED_STATUSES)].copy()
confirmed["__Group"] = "Confirmed / Approved"
pending["__Group"]   = "Preliminary / Pending"

# ----------------- Aggregation & KPIs -----------------
agg_confirmed = aggregate_by_program_status(confirmed, COSTS, "Confirmed / Approved")
agg_pending   = aggregate_by_program_status(pending,   COSTS, "Preliminary / Pending")

conf_students, conf_budget = kpi_row(agg_confirmed)
pend_students, pend_budget = kpi_row(agg_pending)

k1, k2, k3, k4 = st.columns(4)
with k1: st.metric("Approved Student Count", f"{conf_students:,}")
with k2: st.metric("Approved Budget Exposure", fmt_money(conf_budget))
with k3: st.metric("Pending Student Count", f"{pend_students:,}")
with k4: st.metric("Pending Budget Exposure", fmt_money(pend_budget))

# ----------------- Tables -----------------
st.subheader("Approved Students (Committed, Granted, Provisional)")
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
     totals_by(agg_pending,   "Preliminary / Pending")],
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

# ----------------- Download workbook -----------------
out = io.BytesIO()
# Prefer XlsxWriter; install via requirements.txt (xlsxwriter>=3.2)
with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
    if not confirmed.empty: tbl_c.to_excel(writer, index=False, sheet_name="Confirmed_Program_Breakdown")
    if not pending.empty:   tbl_p.to_excel(writer, index=False, sheet_name="Pending_Program_Breakdown")
    if not summary.empty:   summary.to_excel(writer, index=False, sheet_name="Budget_Summary_by_Status")
    pd.DataFrame({"Note": [
        "Unique-student budgeting: per student, the final application is chosen by status priority, then by highest program cost.",
        "Status priority: Committed > Permission to Study Abroad: Granted > Provisional Permission > others.",
        "Budget = Student Count × Program Cost/Student; Pending budget is potential exposure.",
        "Programs without a cost mapping show Budget as NaN."
    ]}).to_excel(writer, index=False, sheet_name="Notes")

st.download_button(
    "Download Budget Workbook (.xlsx)",
    data=out.getvalue(),
    file_name="StudyAbroad_Budget_Dashboard.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)
