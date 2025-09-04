import io
import re
import pandas as pd
import streamlit as st

st.set_page_config(page_title="CEA Budget Planner — Grouped", layout="wide")

st.title("CEA Budget Planner — Grouped (Likely vs Might Go)")
st.markdown(
    "Upload your **Applications export** (tab-delimited .txt) that contains **all students and statuses**, "
    "plus your **Program Cost** file. We'll split into two groups:"
    "\n- **Group A (Likely Going):** Committed, Provisional Permission, Permission to Study Abroad: Granted"
    "\n- **Group B (Might Go):** everyone else (all other statuses), excluding Group A"
)

# ----------------- Helpers -----------------
def read_any_table(file, default_sep="\t"):
    if file is None:
        return None
    try:
        df = pd.read_csv(file, sep=default_sep, dtype=str, engine="python")
    except Exception:
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
    s = str(s)
    s = s.replace(",", "").replace("$", "").strip()
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
    s = re.sub("\s+", " ", s)
    return s

GROUP_A_STATUSES = [
    "Committed",
    "Provisional Permission",
    "Permission to Study Abroad: Granted",
]

# ----------------- Sidebar -----------------
st.sidebar.header("1) Upload Data")
apps_file = st.sidebar.file_uploader("Applications export (.txt/.tsv/.csv)", type=["txt", "tsv", "csv"])
costs_file = st.sidebar.file_uploader("Program costs file", type=["txt", "tsv", "csv"])

if apps_file is None:
    st.info("Please upload the Applications export to begin.")
    st.stop()

apps = read_any_table(apps_file, default_sep="\t")

# Map expected columns, based on the user's export
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
    st.error(f"Missing expected columns in Applications file: {missing}. "
             "Please verify your export headers.")
    st.stop()

# Normalize key fields
apps["__Program"] = apps[resolved["Program_Name"]].map(coalesce_program_keys)
apps["__Status"] = apps[resolved["Application_Status"]].fillna("")
apps["__Term"] = apps[resolved["Program_Term"]] if "Program_Term" in resolved else ""
apps["__Year"] = apps[resolved["Program_Year"]] if "Program_Year" in resolved else ""

# Filters
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

# Split into Group A and Group B
group_a = filtered[filtered["__Status"].isin(GROUP_A_STATUSES)].copy()
group_b = filtered[~filtered["__Status"].isin(GROUP_A_STATUSES)].copy()
group_a["__Group"] = "Likely Going (A)"
group_b["__Group"] = "Might Go (B)"

# Costs (optional but recommended)
if costs_file is not None:
    costs = read_any_table(costs_file, default_sep="\t")
    cost_name_col = None
    for c in ["Program Name", "Program_Name", "Program", "Name"]:
        if c in costs.columns:
            cost_name_col = c
            break
    cost_value_col = None
    for c in ["$ Cost/Student", "Cost", "Cost per Student", "Cost_Student"]:
        if c in costs.columns:
            cost_value_col = c
            break
    if cost_name_col and cost_value_col:
        costs["__Program"] = costs[cost_name_col].map(coalesce_program_keys)
        costs["__Cost"] = costs[cost_value_col].map(normalize_currency)
        costs = costs[["__Program", "__Cost"]]
    else:
        st.warning("Could not detect Program/Cost columns in the cost file. Budget totals will be shown without costs.")
        costs = pd.DataFrame(columns=["__Program", "__Cost"])
else:
    costs = pd.DataFrame(columns=["__Program", "__Cost"])

def aggregate_by_program_status(df):
    if df.empty:
        return pd.DataFrame(columns=["__Program", "__Status", "Students", "__Cost", "Program Cost/Student (USD)", "Budget", "__Group"])
    counts = (
        df.groupby(["__Program", "__Status"], dropna=False)
          .size()
          .reset_index(name="Students")
    )
    merged = counts.merge(costs, on="__Program", how="left")
    merged["Program Cost/Student (USD)"] = merged["__Cost"]
    merged["Budget"] = merged["Students"] * merged["__Cost"]
    if "__Group" in df.columns:
        merged["__Group"] = df["__Group"].iloc[0]
    else:
        merged["__Group"] = ""
    return merged

agg_a = aggregate_by_program_status(group_a)
agg_b = aggregate_by_program_status(group_b)

# KPIs
def kpi_row(df, label):
    total_students = df["Students"].sum() if not df.empty else 0
    total_budget = df["Budget"].sum(min_count=1) if "Budget" in df else pd.NA
    return label, total_students, total_budget

kpi_a = kpi_row(agg_a, "Group A — Likely Going")
kpi_b = kpi_row(agg_b, "Group B — Might Go (Potential)")

def fmt_money(x):
    return "-" if pd.isna(x) else f"${x:,.0f}"

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Group A Students", f"{kpi_a[1]:,}")
with col2:
    st.metric("Group A Budget", fmt_money(kpi_a[2]))
with col3:
    st.metric("Group B Students", f"{kpi_b[1]:,}")
with col4:
    st.metric("Group B Potential Budget", fmt_money(kpi_b[2]))

# Tables
st.subheader("Group A — Likely Going (Committed / Provisional / Granted)")
if agg_a.empty:
    st.write("No rows in Group A for current filters.")
else:
    pretty_a = agg_a.rename(columns={"__Program": "Program", "__Status": "Status", "Students": "Student Count"})
    st.dataframe(pretty_a.sort_values(["Program", "Status"]), use_container_width=True)

st.subheader("Group B — Might Go (All Other Statuses)")
if agg_b.empty:
    st.write("No rows in Group B for current filters.")
else:
    pretty_b = agg_b.rename(columns={"__Program": "Program", "__Status": "Status", "Students": "Student Count"})
    st.dataframe(pretty_b.sort_values(["Program", "Status"]), use_container_width=True)

# Totals by group and status
def totals_by(df, label):
    if df.empty:
        return pd.DataFrame(columns=["Group", "Status", "Students", "Budget"])
    t = (
        df.groupby("__Status", dropna=False)[["Students", "Budget"]]
          .sum(min_count=1)
          .reset_index()
          .rename(columns={"__Status": "Status"})
    )
    t.insert(0, "Group", label)
    return t

totals = pd.concat([totals_by(agg_a, "A — Likely"), totals_by(agg_b, "B — Might")], ignore_index=True)
st.subheader("Totals by Group and Status")
st.dataframe(totals, use_container_width=True)

# Diagnostics: programs without costs
all_agg = pd.concat([agg_a, agg_b], ignore_index=True)
missing_cost = sorted(all_agg[all_agg["Program Cost/Student (USD)"].isna()]["__Program"].dropna().unique().tolist())
if missing_cost:
    with st.expander("Programs missing cost mapping (click to review)"):
        st.write(missing_cost)

# Download workbook
out = io.BytesIO()
with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
    if not group_a.empty:
        pretty_a.to_excel(writer, index=False, sheet_name="GroupA_ProgramBreakdown")
    if not group_b.empty:
        pretty_b.to_excel(writer, index=False, sheet_name="GroupB_ProgramBreakdown")
    if not totals.empty:
        totals.to_excel(writer, index=False, sheet_name="Totals_by_Group_and_Status")
    notes = pd.DataFrame({
        "Note": [
            "Group A statuses: Committed, Provisional Permission, Permission to Study Abroad: Granted.",
            "Group B includes all other statuses from the Applications export.",
            "Budget = Student Count × Program Cost/Student; Group B budget treated as potential exposure.",
            "Programs without a matching cost will show Budget as NaN.",
        ]
    })
    notes.to_excel(writer, index=False, sheet_name="Notes")

st.download_button(
    "Download Budget Workbook (.xlsx)",
    data=out.getvalue(),
    file_name="CEA_Budget_Planner_Grouped.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)