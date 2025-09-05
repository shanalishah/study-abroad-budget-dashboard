import io
import re
import glob
import json
import hashlib
from pathlib import Path
from datetime import datetime
import pandas as pd
import streamlit as st

try:
    import yaml  # optional; only needed if you use YAML costs
except Exception:
    yaml = None

st.set_page_config(page_title="Study Abroad Budget Dashboard", layout="wide")

st.title("Study Abroad Budget Dashboard")
st.markdown(
    "• Defaults to the latest timestamp-named applications file (e.g., `202594152731.txt`).\n"
    "• Uploading a file overrides the default **for this session**.\n"
    "• Program costs are loaded from **ProgramCost.(tsv|txt|csv|yaml|yml|json)** in the repo.\n"
    "• Budget counts **unique students** (one final app per student): "
    "**Committed** > **Granted** > **Provisional**; ties break to **highest program cost**."
)

# ------------------------------------------------------------
# File discovery
# ------------------------------------------------------------
SUPPORTED_COST_EXTS = [".tsv", ".txt", ".csv", ".yaml", ".yml", ".json"]

def find_cost_file() -> Path | None:
    """Find ProgramCost file; prefer TSV/TXT, then CSV, then YAML/JSON."""
    candidates = []
    for ext in SUPPORTED_COST_EXTS:
        p = Path(f"ProgramCost{ext}")
        if p.exists():
            candidates.append(p)
    preference = {".tsv": 0, ".txt": 0, ".csv": 1, ".yaml": 2, ".yml": 2, ".json": 3}
    candidates.sort(key=lambda p: preference.get(p.suffix.lower(), 9))
    return candidates[0] if candidates else None

def find_latest_app_file() -> Path | None:
    """Pick the newest applications file by numeric value of its timestamp stem (e.g., 202594152731.txt)."""
    ts_re = re.compile(r"^(\d{4})(\d{1,2})(\d{1,2})(\d{6})$")
    candidates = []
    for f in glob.glob("*.txt"):
        stem = Path(f).stem
        if stem.isdigit() and ts_re.match(stem):
            candidates.append(Path(f))
    if not candidates:
        return None
    return max(candidates, key=lambda p: int(p.stem))

def parse_timestamp_label(stem: str) -> str:
    ts_re = re.compile(r"^(\d{4})(\d{1,2})(\d{1,2})(\d{6})$")
    m = ts_re.match(stem)
    if not m:
        return stem
    y, mo, d, hms = m.groups()
    try:
        dt = datetime(int(y), int(mo), int(d), int(hms[0:2]), int(hms[2:4]), int(hms[4:6]))
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return stem

# ------------------------------------------------------------
# Encoding-robust readers
# ------------------------------------------------------------
def _read_csv_forgiving(path_or_buf, sep, dtype=str):
    """
    Try encodings in order: utf-8, utf-8-sig (BOM), cp1252.
    Replace undecodable bytes and skip truly broken lines.
    """
    encodings = ["utf-8", "utf-8-sig", "cp1252"]
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(
                path_or_buf,
                sep=sep,
                dtype=dtype,
                engine="python",
                encoding=enc,
                encoding_errors="replace",
                on_bad_lines="skip",
            )
        except Exception as e:
            last_err = e
    raise last_err

def _read_text(path: Path) -> str:
    """Read text with forgiving encodings for YAML/JSON."""
    for enc in ["utf-8", "utf-8-sig", "cp1252"]:
        try:
            return path.read_text(encoding=enc, errors="replace")
        except Exception:
            continue
    return path.read_bytes().decode("utf-8", errors="replace")

def read_any_table(file_or_path, default_sep="\t"):
    """
    Read tab-delimited first; fall back to CSV.
    Works for file-like or Path/str. Trims whitespace.
    Uses forgiving encodings to avoid UnicodeDecodeError.
    """
    if file_or_path is None:
        return None
    # Try tab-delimited
    try:
        df = _read_csv_forgiving(file_or_path, sep=default_sep, dtype=str)
    except Exception:
        # Reset pointer if file-like
        if hasattr(file_or_path, "seek"):
            try:
                file_or_path.seek(0)
            except Exception:
                pass
        # Try comma-delimited
        df = _read_csv_forgiving(file_or_path, sep=",", dtype=str)

    df.columns = [c.strip() for c in df.columns]
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip()
    return df

def normalize_currency(s):
    if pd.isna(s): return pd.NA
    s = str(s).replace(",", "").replace("$", "").strip()
    if s == "" or s.lower() == "na": return pd.NA
    try: return float(s)
    except Exception: return pd.NA

def coalesce_program_keys(name):
    if pd.isna(name): return None
    s = str(name).strip()
    s = re.sub("[\u2013\u2014]", "-", s)  # en/em dash -> hyphen
    s = s.replace("’", "'").replace("“", '"').replace("”", '"')
    s = re.sub(r"\s+", " ", s)
    return s

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
    if not path or not path.exists(): return ""
    return hashlib.md5(path.read_bytes()).hexdigest()

# ------------------------------------------------------------
# Load ProgramCost.* (TSV/TXT/CSV/YAML/JSON) with caching
# ------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=0)
def load_costs_from_repo(path: Path, content_hash: str) -> pd.DataFrame:
    """
    Returns tidy DataFrame with columns: __Program, __Cost.
    Accepts:
      - TSV/TXT: tab-delimited with headers (Program Name, $ Cost/Student)
      - CSV: comma-delimited with headers
      - YAML: list of {program, cost} or {Program Name, $ Cost/Student}
      - JSON: array of objects with program & cost keys
    """
    if not path or not path.exists():
        st.warning("No ProgramCost file found. Expected one of: ProgramCost.tsv/.txt/.csv/.yaml/.yml/.json")
        return pd.DataFrame(columns=["__Program", "__Cost"])

    ext = path.suffix.lower()

    if ext in [".tsv", ".txt"]:
        df = _read_csv_forgiving(path, sep="\t", dtype=str)
    elif ext == ".csv":
        df = _read_csv_forgiving(path, sep=",", dtype=str)
    elif ext in [".yaml", ".yml"]:
        if yaml is None:
            st.error("YAML support requires the 'pyyaml' package. Add it to requirements.txt or use CSV/TSV.")
            return pd.DataFrame(columns=["__Program", "__Cost"])
        txt = _read_text(path)
        data = yaml.safe_load(txt) if txt else []
        df = pd.DataFrame(data)
    elif ext == ".json":
        txt = _read_text(path)
        data = json.loads(txt) if txt else []
        df = pd.DataFrame(data)
    else:
        st.error(f"Unsupported ProgramCost format: {ext}")
        return pd.DataFrame(columns=["__Program", "__Cost"])

    # Normalize headers
    df.columns = [str(c).strip() for c in df.columns]

    # Identify name & cost columns
    name_col = next((c for c in ["Program Name", "Program_Name", "program", "Program", "Name"] if c in df.columns), None)
    cost_col = next((c for c in ["$ Cost/Student", "Cost", "cost", "Cost per Student", "Cost_Student"] if c in df.columns), None)

    if name_col is None or cost_col is None:
        st.warning("ProgramCost file must include columns for Program Name and Cost (e.g., 'Program Name', '$ Cost/Student').")
        return pd.DataFrame(columns=["__Program", "__Cost"])

    # Keep only rows with a program name
    df = df[~df[name_col].isna() & (df[name_col].astype(str).str.strip() != "")].copy()
    df["__Program"] = df[name_col].map(coalesce_program_keys)
    df["__Cost"]    = df[cost_col].map(normalize_currency)
    df = df[~df["__Program"].isna() & (df["__Program"].astype(str).str.strip() != "")]
    return df[["__Program", "__Cost"]]

# ------------------------------------------------------------
# Student dedup logic
# ------------------------------------------------------------
STATUS_PRIORITY = {"Committed": 3, "Permission to Study Abroad: Granted": 2, "Provisional Permission": 1}
def status_rank(s): return STATUS_PRIORITY.get(s, 0)

def build_student_key(df: pd.DataFrame) -> pd.Series:
    candidates = [
        "UR ID", "UR_ID", "Student_ID", "Student Id", "ID",
        "Email", "Email_Address", "Email Address", "Student Email",
        "NetID", "Net Id"
    ]
    found = next((c for c in candidates if c in df.columns), None)
    if found:
        return df[found].astype(str).str.strip().str.lower().replace({"nan": ""})
    fn = df.columns[df.columns.str.lower().str.contains("first")].tolist()
    ln = df.columns[df.columns.str.lower().str.contains("last")].tolist()
    if fn and ln:
        return (df[fn[0]].astype(str).str.strip().str.lower() + "||" +
                df[ln[0]].astype(str).str.strip().str.lower())
    return (df.get("Email", pd.Series([""]*len(df))).astype(str).str.lower() + "||" +
            df.get("__Program", pd.Series([""]*len(df))).astype(str).str.lower())

def dedup_students_by_priority_then_cost(df_apps: pd.DataFrame, df_costs: pd.DataFrame) -> pd.DataFrame:
    if df_apps.empty:
        return df_apps
    costs = df_costs.rename(columns={"__Program": "__Program_cost_key", "__Cost": "__CostValue"}).copy()
    tmp = df_apps.copy()
    tmp["__StudentKey"] = build_student_key(tmp)
    tmp["__Rank"] = tmp["__Status"].map(status_rank)
    tmp = tmp.merge(
        costs[["__Program_cost_key", "__CostValue"]],
        left_on="__Program", right_on="__Program_cost_key", how="left"
    )
    tmp["__CostValue"] = tmp["__CostValue"].fillna(0.0)
    tmp = tmp.sort_values(["__Rank", "__CostValue"], ascending=[False, False], kind="stable")
    tmp = tmp.drop_duplicates(subset="__StudentKey", keep="first")
    return tmp.drop(columns=["__StudentKey", "__Rank", "__Program_cost_key", "__CostValue"], errors="ignore")

def aggregate_by_program_status(df, costs, group_label):
    if df.empty:
        return pd.DataFrame(columns=["__Program","__Status","Students","__Cost","Program Cost/Student (USD)","Budget","__Group"])
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

# ------------------------------------------------------------
# Sidebar controls
# ------------------------------------------------------------
if st.sidebar.button("Refresh data from repository"):
    st.cache_data.clear()
    st.experimental_rerun()

st.sidebar.header("1) Applications Data")
uploaded_apps = st.sidebar.file_uploader("(Optional) Upload applications export", type=["txt","tsv","csv"])

if "use_repo_default" not in st.session_state:
    st.session_state.use_repo_default = True

if uploaded_apps is not None:
    st.session_state.use_repo_default = False

if st.sidebar.button("Use latest repository file"):
    st.session_state.use_repo_default = True

DEFAULT_APPS_PATH = find_latest_app_file()
if st.session_state.use_repo_default:
    if DEFAULT_APPS_PATH:
        st.sidebar.success(f"Using default: {DEFAULT_APPS_PATH.name} ({parse_timestamp_label(DEFAULT_APPS_PATH.stem)})")
    else:
        st.sidebar.warning("No timestamp-named applications file found (e.g., 202594152731.txt).")
else:
    st.sidebar.info("Using uploaded file (this session only)")

# ------------------------------------------------------------
# Load costs & applications
# ------------------------------------------------------------
COST_FILE_PATH = find_cost_file()
COSTS = load_costs_from_repo(COST_FILE_PATH, file_md5(COST_FILE_PATH))

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

# Filters
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

# Unique-student dedup
filtered = dedup_students_by_priority_then_cost(filtered, COSTS)

# Cohorts
CONFIRMED_STATUSES = ["Committed", "Permission to Study Abroad: Granted", "Provisional Permission"]
confirmed = filtered[filtered["__Status"].isin(CONFIRMED_STATUSES)].copy()
pending   = filtered[~filtered["__Status"].isin(CONFIRMED_STATUSES)].copy()
confirmed["__Group"] = "Confirmed / Approved"
pending["__Group"]   = "Preliminary / Pending"

# Aggregation & KPIs
agg_confirmed = aggregate_by_program_status(confirmed, COSTS, "Confirmed / Approved")
agg_pending   = aggregate_by_program_status(pending,   COSTS, "Preliminary / Pending")

conf_students, conf_budget = kpi_row(agg_confirmed)
pend_students, pend_budget = kpi_row(agg_pending)

k1, k2, k3, k4 = st.columns(4)
with k1: st.metric("Confirmed Student Count", f"{conf_students:,}")
with k2: st.metric("Confirmed Budget Exposure", fmt_money(conf_budget))
with k3: st.metric("Pending Student Count", f"{pend_students:,}")
with k4: st.metric("Pending Budget Exposure", fmt_money(pend_budget))

# Tables
st.subheader("Confirmed / Approved Students (Committed, Granted, Provisional)")
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

# Summary
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

# Diagnostics
all_agg = pd.concat([agg_confirmed, agg_pending], ignore_index=True)
missing_cost = sorted(all_agg[all_agg["Program Cost/Student (USD)"].isna()]["__Program"].dropna().unique().tolist())
if missing_cost:
    with st.expander("Programs missing cost mapping (click to review)"):
        st.write(missing_cost)

# Download workbook
out = io.BytesIO()
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
