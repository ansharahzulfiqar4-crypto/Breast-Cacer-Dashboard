"""Interactive, descriptive dashboard for the Breast Cancer dataset."""

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(
    page_title="Breast Cancer Dataset | Explorer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_PATH = Path(__file__).resolve().parent / "Breast_Cancer.csv"
COLORS = {"Alive": "#168C82", "Dead": "#D45A66"}
PALETTE = px.colors.qualitative.Safe


def load_data(uploaded_file=None):
    """Load and normalize the project CSV or an uploaded CSV."""
    if uploaded_file is not None:
        frame = pd.read_csv(uploaded_file)
    elif DATA_PATH.exists():
        frame = pd.read_csv(DATA_PATH)
    else:
        return None

    frame.columns = (
        frame.columns.astype(str).str.strip().str.lower()
        .str.replace(r"\s+", "_", regex=True)
    )
    # Trim whitespace around categorical labels while keeping missing values intact.
    for column in frame.select_dtypes(include=["object", "string"]).columns:
        frame[column] = frame[column].map(lambda value: value.strip() if isinstance(value, str) else value)
    for column in frame.columns:
        if column in {"age", "grade", "tumor_size", "regional_node_examined", "reginol_node_positive", "survival_months"}:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if "tumor_size" in frame.columns:
        frame["tumor_size_category"] = pd.cut(
            frame["tumor_size"], bins=[0, 20, 50, float("inf")],
            labels=["Small", "Medium", "Large"], include_lowest=True,
        )
    return frame


def metric_card(label, value, detail=None):
    st.metric(label, value, help=detail)


def numeric_columns(frame):
    return frame.select_dtypes(include="number").columns.tolist()


def categorical_columns(frame):
    return [c for c in frame.columns if c not in numeric_columns(frame)]


def apply_filters(frame):
    filtered = frame.copy()
    st.sidebar.markdown("### Dataset filters")
    status_col = "status" if "status" in frame.columns else None
    if status_col:
        values = sorted(frame[status_col].dropna().unique().tolist())
        chosen = st.sidebar.multiselect("Patient status", values, default=values)
        filtered = filtered[filtered[status_col].isin(chosen)]
    for column, label in [("race", "Race"), ("6th_stage", "6th stage"), ("a_stage", "A stage")]:
        if column in frame.columns:
            values = sorted(frame[column].dropna().unique().tolist(), key=str)
            chosen = st.sidebar.multiselect(label, values, default=values, key=f"filter_{column}")
            filtered = filtered[filtered[column].isin(chosen)]
    return filtered


def chart_layout(fig, height=420):
    fig.update_layout(
        height=height, margin=dict(l=12, r=12, t=52, b=12),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#26364A"),
        legend_title_text="",
    )
    fig.update_xaxes(showgrid=False, linecolor="#E6EBF1")
    fig.update_yaxes(gridcolor="#EEF1F5", zeroline=False)
    return fig


st.markdown("""
<style>
    .stApp { background: #F5F7FA; }
    [data-testid="stSidebar"] { background: #101D31; }
    [data-testid="stSidebar"] * { color: #EAF0F7; }
    [data-testid="stMetric"] { background: white; border: 1px solid #E7EBF0;
        padding: 17px 19px; border-radius: 13px; box-shadow: 0 2px 8px #182a4208; }
    [data-testid="stMetricLabel"] { color: #65748A; }
    [data-testid="stMetricValue"] { color: #172A43; }
    .hero { padding: 25px 28px; border-radius: 16px; color: white;
        background: linear-gradient(115deg,#12243B,#1A5262); margin-bottom: 20px; }
    .hero h1 { color:white; margin:0 0 6px 0; font-size:2rem; }
    .hero p { color:#D6E7ED; margin:0; }
    .section-note { color:#64748B; font-size:.93rem; }
    div[data-testid="stPlotlyChart"] { background:white; border:1px solid #E7EBF0;
        border-radius:14px; padding:8px; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## Dataset Explorer")
    st.caption("Breast cancer records · descriptive analysis")
    uploaded = st.file_uploader("Use another CSV", type=["csv"], help="By default, the app loads Breast_Cancer.csv beside app.py.")

df = load_data(uploaded)
if df is None:
    st.error(f"Could not find the dataset at `{DATA_PATH}`. Place Breast_Cancer.csv beside app.py or upload a CSV in the sidebar.")
    st.stop()

filtered = apply_filters(df)
pages = ["Overview", "Data analysis", "Univariate analysis", "Bivariate analysis", "Multivariate analysis", "Business insights"]
page = st.sidebar.radio("Navigate", pages, label_visibility="visible")
st.sidebar.divider()
st.sidebar.caption(f"Showing {len(filtered):,} of {len(df):,} records")
st.sidebar.download_button(
    "Download filtered CSV", filtered.to_csv(index=False).encode("utf-8"),
    "breast_cancer_filtered.csv", "text/csv", use_container_width=True,
)

st.markdown('<div class="hero"><h1>Breast Cancer Dataset Explorer</h1><p>Explore the dataset’s recorded patient, tumor, and outcome characteristics.</p></div>', unsafe_allow_html=True)
st.caption("Descriptive dataset exploration only. These records and comparisons do not diagnose cancer or predict an individual’s outcome.")

if page == "Overview":
    st.header("Overview")
    st.write("A filtered snapshot of the source dataset and its main recorded measures.")
    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("Records in view", f"{len(filtered):,}")
    with c2: metric_card("Dataset columns", f"{len(df.columns):,}")
    with c3: metric_card("Missing cells", f"{int(filtered.isna().sum().sum()):,}")
    with c4: metric_card("Duplicate rows", f"{int(filtered.duplicated().sum()):,}")
    st.subheader("Status distribution")
    if filtered.empty:
        st.info("No records match the current filters.")
    elif "status" in filtered:
        counts = filtered["status"].value_counts().rename_axis("status").reset_index(name="records")
        fig = px.pie(counts, names="status", values="records", hole=.62,
                     color="status", color_discrete_map=COLORS, title="Recorded status in the filtered data")
        fig.update_traces(textposition="inside", textinfo="label+percent")
        st.plotly_chart(chart_layout(fig, 360), use_container_width=True)
        st.dataframe(counts.assign(share=counts.records / len(filtered)).style.format({"share":"{:.1%}"}), hide_index=True, use_container_width=True)
    st.subheader("Key findings from the full dataset")
    if "status" in df and df["status"].notna().any():
        shares = df["status"].value_counts(normalize=True)
        lines = [f"- **{name}:** {int(df['status'].eq(name).sum()):,} records ({share:.1%})." for name, share in shares.items()]
        st.markdown("\n".join(lines))
    if {"status", "age", "tumor_size", "survival_months"}.issubset(df.columns):
        means = df.groupby("status")[["age", "tumor_size", "survival_months"]].mean()
        if len(means) > 1:
            st.markdown("Group averages in this dataset: " + "; ".join(
                f"**{group}** — age {row.age:.1f}, tumor size {row.tumor_size:.1f}, recorded survival months {row.survival_months:.1f}"
                for group, row in means.iterrows()) + ".")

elif page == "Data analysis":
    st.header("Data analysis & quality")
    left, right = st.columns([1.5, 1])
    with left:
        st.subheader("Dataset preview")
        st.dataframe(filtered.head(100), use_container_width=True, hide_index=True)
        st.caption("Preview is limited to the first 100 rows in the filtered view.")
    with right:
        st.subheader("Column profile")
        profile = pd.DataFrame({"Column": df.columns, "Data type": df.dtypes.astype(str).values,
            "Missing": df.isna().sum().values, "Missing %": (df.isna().mean().values * 100).round(2),
            "Unique": df.nunique(dropna=True).values})
        st.dataframe(profile, use_container_width=True, hide_index=True, height=520)
    missing = df.isna().sum()
    st.subheader("Quality summary")
    q1,q2,q3 = st.columns(3)
    with q1: metric_card("Exact duplicate rows", f"{int(df.duplicated().sum()):,}")
    with q2: metric_card("Cells missing", f"{int(missing.sum()):,}")
    with q3: metric_card("Rows with any missing value", f"{int(df.isna().any(axis=1).sum()):,}")
    st.info("Duplicate rows are reported, not silently removed. The category field `tumor_size_category` is derived from `tumor_size` using the source notebook’s bins (0–20, >20–50, >50).")
    st.subheader("Numerical summary")
    nums = numeric_columns(df)
    if nums: st.dataframe(filtered[nums].describe().T, use_container_width=True)

elif page == "Univariate analysis":
    st.header("Univariate analysis")
    if filtered.empty:
        st.info("No records match the current filters.")
    else:
        numeric = numeric_columns(filtered)
        cats = categorical_columns(filtered)
        left,right = st.columns(2)
        with left:
            st.subheader("Numerical distribution")
            if numeric:
                col = st.selectbox("Choose a numerical field", numeric, key="uni_num")
                fig = px.histogram(filtered, x=col, nbins=30, marginal="box", color_discrete_sequence=["#31899A"], title=f"Distribution of {col.replace('_',' ').title()}")
                st.plotly_chart(chart_layout(fig), use_container_width=True)
                st.dataframe(filtered[col].describe().to_frame("value").T, use_container_width=True)
            else: st.info("No numerical fields available.")
        with right:
            st.subheader("Category counts")
            if cats:
                col = st.selectbox("Choose a categorical field", cats, key="uni_cat")
                counts = filtered[col].fillna("Missing").value_counts().rename_axis(col).reset_index(name="records")
                fig = px.bar(counts, x=col, y="records", color=col, color_discrete_sequence=PALETTE,
                             title=f"Records by {col.replace('_',' ').title()}")
                fig.update_layout(showlegend=False)
                st.plotly_chart(chart_layout(fig), use_container_width=True)
                st.dataframe(counts, use_container_width=True, hide_index=True)
            else: st.info("No categorical fields available.")

elif page == "Bivariate analysis":
    st.header("Bivariate analysis")
    if filtered.empty: st.info("No records match the current filters.")
    else:
        st.markdown('<p class="section-note">Choose variables to compare. Plots show observed associations and do not imply causation.</p>', unsafe_allow_html=True)
        nums = numeric_columns(filtered); cats = categorical_columns(filtered)
        l,r=st.columns(2)
        with l:
            if len(nums)>=2:
                x=st.selectbox("X variable", nums, index=nums.index("age") if "age" in nums else 0)
                yopts=[n for n in nums if n!=x]
                y=st.selectbox("Y variable", yopts, index=yopts.index("survival_months") if "survival_months" in yopts else 0)
                color="status" if "status" in filtered else None
                fig=px.scatter(filtered,x=x,y=y,color=color,opacity=.7,color_discrete_map=COLORS,title=f"{x.replace('_',' ').title()} vs {y.replace('_',' ').title()}",hover_data=[c for c in ["tumor_size_category","race"] if c in filtered])
                st.plotly_chart(chart_layout(fig),use_container_width=True)
        with r:
            if cats and nums:
                cat=st.selectbox("Category for comparison", cats, index=cats.index("status") if "status" in cats else 0)
                val=st.selectbox("Measure", nums, index=nums.index("tumor_size") if "tumor_size" in nums else 0)
                grouped=filtered.groupby(cat, dropna=False)[val].agg(["mean","median","count"]).reset_index()
                grouped[cat]=grouped[cat].fillna("Missing").astype(str)
                fig=px.bar(grouped,x=cat,y="mean",color=cat,color_discrete_sequence=PALETTE,error_y=None,title=f"Average {val.replace('_',' ')} by {cat.replace('_',' ')}",labels={"mean":f"Mean {val.replace('_',' ')}"})
                st.plotly_chart(chart_layout(fig),use_container_width=True)
                st.dataframe(grouped,hide_index=True,use_container_width=True)

elif page == "Multivariate analysis":
    st.header("Multivariate analysis")
    nums=numeric_columns(filtered)
    if len(nums)<2 or filtered.empty: st.info("At least two numerical fields and one matching record are required.")
    else:
        corr=filtered[nums].corr(numeric_only=True)
        fig=px.imshow(corr,text_auto=".2f",color_continuous_scale="RdBu_r",zmin=-1,zmax=1,aspect="auto",title="Correlation between numerical fields")
        st.plotly_chart(chart_layout(fig,560),use_container_width=True)
        st.caption("Pearson correlations describe linear association among numerical fields; they do not establish cause or clinical significance.")
        if {"age","tumor_size","survival_months","status"}.issubset(filtered.columns):
            st.subheader("Age, tumor size & recorded months")
            fig=px.scatter(filtered,x="age",y="tumor_size",size="survival_months",color="status",color_discrete_map=COLORS,opacity=.7,hover_data=["survival_months","6th_stage"] if "6th_stage" in filtered else ["survival_months"],title="Observed records across age and tumor size")
            st.plotly_chart(chart_layout(fig),use_container_width=True)

else:
    st.header("Business insights")
    st.markdown("Questions answered from the available records. Differences are descriptive and may reflect the dataset’s composition; no causal or individual prediction claims are made.")
    if filtered.empty:
        st.info("No records match the current filters.")
    else:
        if "status" in filtered:
            st.subheader("How is recorded status distributed?")
            tab=filtered["status"].value_counts(dropna=False).rename_axis("Status").to_frame("Records")
            tab["Share"]=(tab["Records"]/len(filtered)).map(lambda x:f"{x:.1%}")
            st.dataframe(tab,use_container_width=True)
        if {"status","age","tumor_size","survival_months"}.issubset(filtered.columns):
            st.subheader("How do recorded averages differ by status?")
            grouped=filtered.groupby("status")[["age","tumor_size","survival_months"]].agg(["mean","median","count"])
            st.dataframe(grouped,use_container_width=True)
            if "Alive" in filtered.status.unique() and "Dead" in filtered.status.unique():
                means=filtered.groupby("status")[["age","tumor_size","survival_months"]].mean()
                for field,label in [("age","age"),("tumor_size","tumor size"),("survival_months","recorded survival months")]:
                    diff=means.loc["Dead",field]-means.loc["Alive",field]
                    st.markdown(f"- **{label.title()}:** the mean is {abs(diff):.1f} higher for records labeled Dead." if diff>=0 else f"- **{label.title()}:** the mean is {abs(diff):.1f} higher for records labeled Alive.")
        if {"tumor_size_category","status"}.issubset(filtered.columns):
            st.subheader("Does the status mix vary across tumor size categories?")
            cross=pd.crosstab(filtered["tumor_size_category"],filtered["status"],normalize="index").mul(100).round(1)
            st.dataframe(cross.style.format("{:.1f}%"),use_container_width=True)
            st.caption("Percentages are within each tumor size category; category boundaries follow the source analysis.")
    st.info("Use these summaries to guide further data review. The dataset does not contain treatment, follow-up design, or model validation details needed for clinical recommendations.")

st.divider()
st.caption("Source: Breast_Cancer.csv bundled with this project. Column `reginol_node_positive` retains the source spelling.")
