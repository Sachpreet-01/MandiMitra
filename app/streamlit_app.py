"""MandiMitra: historical agricultural market intelligence (Streamlit app).

The historical Parquet file is queried with DuckDB. It is never loaded
in full into Pandas, which keeps deployment memory use manageable.
"""

from __future__ import annotations

from pathlib import Path

import altair as alt
import duckdb
import numpy as np
import pandas as pd
import streamlit as st

alt.data_transformers.disable_max_rows()

# ----------------------------------------------------------------------
# Paths and constants
# ----------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "app" / "market_comparison_app.parquet"
INFERENCE_PATH = ROOT / "data" / "inference" / "mandimitra_historical_market_rankings.parquet"

HISTORY_COLUMNS = [
    "Arrival_Date", "Commodity", "Variety", "Grade", "State", "District",
    "Market", "Modal_Price", "Peer_Median_Price", "Peer_Market_Count",
    "Premium_Discount_Pct",
]
CATEGORY_COLUMNS = ["Commodity", "Variety", "Grade", "State", "District", "Market"]

INFERENCE_REQUIRED_COLUMNS = {
    "Arrival_Date", "Commodity", "Variety", "Grade", "State", "District", "Market",
    "Eligible_For_Ranking", "Comparison_Group_Size", "Predicted_Rank",
    "Predicted_Premium_Discount_Pct", "Lower_Premium_Discount_Pct",
    "Upper_Premium_Discount_Pct", "Modal_Price", "Peer_Median_Price",
    "Peer_Market_Count",
}

MIN_DAYS_PER_YEAR = 60
MIN_QUALIFYING_YEARS = 3
MIN_COMPARISON_GROUP = 5
STALE_PRICE_DAYS = 30

GREEN = "#2f7d57"
GREY = "#9aa5b1"
RED = "#c0504d"

FLAG_LABELS = {
    "Flag_Extreme_Current_Ratio": "Price far from peers",
    "Flag_High_Volatility": "Volatile",
    "Flag_Long_Reporting_Gap": "Long reporting gap",
    "Flag_Low_Peer_Coverage": "Few peers",
    "Flag_Wide_Prediction_Interval": "Wide interval",
}


# ----------------------------------------------------------------------
# Data access
# ----------------------------------------------------------------------

@st.cache_resource(show_spinner="Connecting to historical market data...")
def get_history_connection():
    """Create one in-memory DuckDB connection over the read-only Parquet source."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Historical data file not found: {DATA_PATH}")

    # Escape the path before embedding it in the read_parquet SQL expression.
    parquet_path = str(DATA_PATH.resolve()).replace("\\", "/").replace("'", "''")
    con = duckdb.connect(database=":memory:")
    con.execute("SET threads = 2")
    con.execute("SET memory_limit = '1200MB'")
    con.execute(
        f"""
        CREATE VIEW historical_prices AS
        SELECT
            CAST(Arrival_Date AS DATE) AS Arrival_Date,
            Commodity, Variety, Grade, State, District, Market,
            Modal_Price, Peer_Median_Price, Peer_Market_Count,
            Premium_Discount_Pct
        FROM read_parquet('{parquet_path}')
        """
    )
    return con


def run_history_query(sql: str, params=(), *, as_dataframe: bool = True):
    """Run one query using a cursor dedicated to this call/session."""
    cursor = get_history_connection().cursor()
    try:
        cursor.execute(sql, list(params))
        return cursor.fetchdf() if as_dataframe else cursor.fetchall()
    finally:
        cursor.close()


@st.cache_data(show_spinner="Loading model rankings...")
def load_inference() -> pd.DataFrame:
    if not INFERENCE_PATH.exists():
        raise FileNotFoundError(f"Inference file not found: {INFERENCE_PATH}")

    frame = pd.read_parquet(INFERENCE_PATH)
    missing = sorted(INFERENCE_REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(
            "The inference dataset is missing required columns: " + ", ".join(missing)
        )
    if frame.empty:
        raise ValueError("The inference dataset contains no records.")

    frame["Arrival_Date"] = pd.to_datetime(frame["Arrival_Date"], errors="coerce")
    if frame["Arrival_Date"].isna().any():
        raise ValueError("The inference dataset contains missing or invalid observation dates.")
    return frame


@st.cache_data(show_spinner=False)
def get_data_end() -> pd.Timestamp:
    rows = run_history_query(
        "SELECT MAX(Arrival_Date) AS data_end FROM historical_prices",
        as_dataframe=False,
    )
    if not rows or rows[0][0] is None:
        raise ValueError("The historical dataset contains no valid arrival dates.")
    return pd.Timestamp(rows[0][0])


@st.cache_data(show_spinner=False)
def get_options(column: str, filters: tuple = ()) -> list[str]:
    """Get distinct dropdown values after applying earlier cascade choices."""
    if column not in CATEGORY_COLUMNS:
        raise ValueError(f"Unsupported category column: {column}")
    where, params = build_where(filters)
    predicate = f"{where} AND" if where else "WHERE"
    query = f"""
        SELECT DISTINCT CAST({column} AS VARCHAR) AS value
        FROM historical_prices
        {predicate} {column} IS NOT NULL
        ORDER BY value
    """
    rows = run_history_query(query, params, as_dataframe=False)
    return [row[0] for row in rows]


def build_where(filters: tuple):
    """Build parameterized SQL predicates from (column, value) pairs."""
    clauses = []
    params = []
    allowed = set(CATEGORY_COLUMNS)
    for column, value in filters:
        if column not in allowed:
            raise ValueError(f"Unsupported filter column: {column}")
        clauses.append(f"{column} = ?")
        params.append(value)
    return ("WHERE " + " AND ".join(clauses) if clauses else ""), params


@st.cache_data(show_spinner=False)
def query_market_history(filters: tuple) -> pd.DataFrame:
    where, params = build_where(filters)
    sql = f"""
        SELECT {", ".join(HISTORY_COLUMNS)}
        FROM historical_prices
        {where}
        ORDER BY Arrival_Date
    """
    frame = run_history_query(sql, params)
    if not frame.empty:
        frame["Arrival_Date"] = pd.to_datetime(frame["Arrival_Date"])
    return frame


@st.cache_data(show_spinner=False)
def query_latest_prices(commodity: str, variety: str, grade: str) -> pd.DataFrame:
    """Latest reported modal price per market for one commodity/variety/grade."""
    sql = """
        SELECT State, District, Market, Arrival_Date, Modal_Price
        FROM historical_prices
        WHERE Commodity = ? AND Variety = ? AND Grade = ?
        QUALIFY ROW_NUMBER() OVER (
            PARTITION BY State, District, Market
            ORDER BY Arrival_Date DESC
        ) = 1
        ORDER BY Market, District, State
    """
    frame = run_history_query(sql, [commodity, variety, grade])
    if frame.empty:
        return frame
    frame["Arrival_Date"] = pd.to_datetime(frame["Arrival_Date"])
    for col in ["State", "District", "Market"]:
        frame[col] = frame[col].astype(str)
    frame["Label"] = frame["Market"] + " | " + frame["District"] + ", " + frame["State"]
    return frame.sort_values("Label").reset_index(drop=True)


# ----------------------------------------------------------------------
# Small UI helpers
# ----------------------------------------------------------------------

def show_table(frame: pd.DataFrame, **kwargs) -> None:
    try:
        st.dataframe(frame, width="stretch", hide_index=True, **kwargs)
    except (TypeError, ValueError):
        st.dataframe(frame, use_container_width=True, hide_index=True, **kwargs)


def show_chart(chart: alt.Chart) -> None:
    try:
        st.altair_chart(chart, width="stretch")
    except (TypeError, ValueError):
        st.altair_chart(chart, use_container_width=True)


def pct(value: float, signed: bool = True) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value:+.2f}%" if signed else f"{value:.1f}%"


def rupees(value: float, decimals: int = 0) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"₹{value:,.{decimals}f}"


def warning_text(frame: pd.DataFrame) -> pd.Series:
    present = [c for c in FLAG_LABELS if c in frame.columns]
    if not present:
        return pd.Series("", index=frame.index)
    parts = [
        np.where(frame[c].fillna(False).astype(bool), FLAG_LABELS[c], "")
        for c in present
    ]
    joined = ["; ".join(p for p in row if p) for row in zip(*parts)]
    return pd.Series(joined, index=frame.index)


def csv_button(frame: pd.DataFrame, label: str, filename: str, key: str) -> None:
    st.download_button(
        label,
        frame.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
        key=key,
    )


def history_cascade(specs: list[tuple[str, str]], prefix: str):
    """Create cascading filters using DuckDB distinct-value queries."""
    filters = []
    selections = {}
    for column, label in specs:
        options = get_options(column, tuple(filters))
        if not options:
            return filters, selections, False
        choice = st.sidebar.selectbox(label, options, key=f"{prefix}_{column}")
        filters.append((column, choice))
        selections[column] = choice
    return filters, selections, True


def option_list(frame: pd.DataFrame, column: str) -> list:
    return sorted(frame[column].dropna().unique().tolist())


# ----------------------------------------------------------------------
# Page 1: historical market explorer
# ----------------------------------------------------------------------

def annual_summary(market_df: pd.DataFrame) -> pd.DataFrame:
    comp = market_df.dropna(subset=["Premium_Discount_Pct"]).copy()
    if comp.empty:
        return pd.DataFrame()
    comp["Year"] = comp["Arrival_Date"].dt.year
    grouped = comp.groupby("Year")
    out = grouped.agg(
        Active_Days=("Arrival_Date", "nunique"),
        Median_Price=("Modal_Price", "median"),
        Median_Premium_Pct=("Premium_Discount_Pct", "median"),
    )
    out["Days_Above_Peer_Pct"] = grouped["Premium_Discount_Pct"].apply(
        lambda s: (s > 0).mean() * 100
    )
    out["Qualifies"] = out["Active_Days"] >= MIN_DAYS_PER_YEAR
    return out.reset_index()


def consistency_label(annual: pd.DataFrame) -> tuple[str, str]:
    if annual.empty:
        return "No data", "There are no comparison days for this selection."
    qualifying = annual[annual["Qualifies"]]
    if len(qualifying) < MIN_QUALIFYING_YEARS:
        return (
            "Not enough history",
            f"Needs at least {MIN_QUALIFYING_YEARS} years with "
            f"{MIN_DAYS_PER_YEAR}+ reporting days; this market has {len(qualifying)}.",
        )
    share_positive = (qualifying["Median_Premium_Pct"] > 0).mean()
    median_premium = qualifying["Median_Premium_Pct"].median()
    if share_positive >= 0.75 and median_premium > 0:
        return "Consistently above peers", f"{share_positive:.0%} of qualifying years had a positive median premium."
    if share_positive <= 0.25 and median_premium < 0:
        return "Consistently below peers", f"Only {share_positive:.0%} of qualifying years had a positive median premium."
    return "Mixed", f"{share_positive:.0%} of qualifying years had a positive median premium."


def render_explorer() -> None:
    st.header("Historical market explorer")
    st.write(
        "Pick a market and see how its modal price compared with the median "
        "of other markets reporting the same commodity, variety and grade "
        "on the same day."
    )

    st.sidebar.subheader("Market")
    filters, sel, complete = history_cascade(
        [
            ("Commodity", "Commodity"), ("Variety", "Variety"),
            ("Grade", "Grade"), ("State", "State"),
            ("District", "District"), ("Market", "Market"),
        ],
        "exp",
    )
    if not complete:
        st.info("No records for this selection.")
        return

    # This is one selected market's time series, not the full national dataset.
    market_df = query_market_history(tuple(filters))
    if market_df.empty:
        st.info("No records for this selection.")
        return

    first, last = market_df["Arrival_Date"].min(), market_df["Arrival_Date"].max()
    if first < last:
        start, end = st.slider(
            "Date range", min_value=first.date(), max_value=last.date(),
            value=(first.date(), last.date()), format="DD MMM YYYY",
        )
        market_df = market_df[
            (market_df["Arrival_Date"] >= pd.Timestamp(start))
            & (market_df["Arrival_Date"] < pd.Timestamp(end) + pd.Timedelta(days=1))
        ]
    if market_df.empty:
        st.info("No records in this date range.")
        return

    comp = market_df.dropna(subset=["Premium_Discount_Pct", "Peer_Median_Price"])
    dates = market_df["Arrival_Date"].drop_duplicates().sort_values()
    gaps = dates.diff().dt.days.dropna()

    st.subheader(f"{sel['Market']}, {sel['District']} | {sel['Commodity']}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Reporting days", f"{dates.nunique():,}")
    c2.metric("Median modal price", rupees(market_df["Modal_Price"].median()))
    c3.metric("Median premium vs peers", pct(comp["Premium_Discount_Pct"].median()) if len(comp) else "n/a")
    c4.metric(
        "Days above peer median",
        pct((comp["Premium_Discount_Pct"] > 0).mean() * 100, signed=False) if len(comp) else "n/a",
    )

    c5, c6, c7 = st.columns(3)
    c5.metric("Typical gap between reports", f"{gaps.median():.0f} days" if len(gaps) else "n/a")
    c6.metric("Longest gap", f"{gaps.max():.0f} days" if len(gaps) else "n/a")
    c7.metric("Last report", dates.max().strftime("%d %b %Y"))

    annual = annual_summary(market_df)
    label, detail = consistency_label(annual)
    st.info(
        f"**Historical pattern: {label}.** {detail} This describes the past "
        "only. It does not predict future prices."
    )

    tab_trend, tab_premium, tab_years, tab_data = st.tabs(
        ["Price trend", "Premium over time", "Year by year", "Daily data"]
    )
    with tab_trend:
        long = (
            market_df[["Arrival_Date", "Modal_Price", "Peer_Median_Price"]]
            .rename(columns={"Modal_Price": "Selected market", "Peer_Median_Price": "Peer median"})
            .melt("Arrival_Date", var_name="Series", value_name="Price")
            .dropna()
        )
        chart = (
            alt.Chart(long).mark_line(point=alt.OverlayMarkDef(size=14))
            .encode(
                x=alt.X("Arrival_Date:T", title=None),
                y=alt.Y("Price:Q", title="₹ per quintal"),
                color=alt.Color(
                    "Series:N",
                    scale=alt.Scale(domain=["Selected market", "Peer median"], range=[GREEN, GREY]),
                    legend=alt.Legend(orient="top", title=None),
                ),
                tooltip=[
                    alt.Tooltip("Arrival_Date:T", title="Date"),
                    alt.Tooltip("Series:N"),
                    alt.Tooltip("Price:Q", format=",.0f", title="₹/quintal"),
                ],
            ).interactive(bind_y=False)
        )
        show_chart(chart)
        st.caption(
            "Dots are actual reports, so empty stretches are real reporting gaps, not interpolated prices. "
            "The peer median excludes the selected market and needs at least three other reporters."
        )

    with tab_premium:
        if comp.empty:
            st.info("Peer comparisons are unavailable for the selected dates.")
        else:
            clip = st.checkbox("Clip the chart at ±100%", value=True, key="exp_clip")
            scale = alt.Scale(domain=[-100, 100], clamp=True) if clip else alt.Scale()
            plot = comp[["Arrival_Date", "Premium_Discount_Pct"]].copy()
            plot["Direction"] = np.where(plot["Premium_Discount_Pct"] >= 0, "Above peers", "Below peers")
            bars = alt.Chart(plot).mark_bar().encode(
                x=alt.X("Arrival_Date:T", title=None),
                y=alt.Y("Premium_Discount_Pct:Q", title="Premium vs peer median (%)", scale=scale),
                color=alt.Color(
                    "Direction:N",
                    scale=alt.Scale(domain=["Above peers", "Below peers"], range=[GREEN, RED]),
                    legend=alt.Legend(orient="top", title=None),
                ),
                tooltip=[
                    alt.Tooltip("Arrival_Date:T", title="Date"),
                    alt.Tooltip("Premium_Discount_Pct:Q", format="+.2f", title="Premium %"),
                ],
            )
            zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="#555").encode(y="y:Q")
            show_chart((bars + zero).interactive(bind_y=False))
            st.caption(
                "Premium = (market price / peer median - 1) x 100. A premium is a price difference, not profit. "
                "Grade, unit and reporting practice can all create apparent premiums."
            )

    with tab_years:
        if annual.empty:
            st.info("No comparison days to summarise.")
        else:
            view = annual.rename(columns={
                "Active_Days": "Reporting days",
                "Median_Price": "Median price (₹/quintal)",
                "Median_Premium_Pct": "Median premium (%)",
                "Days_Above_Peer_Pct": "Days above peers (%)",
                "Qualifies": f"Counts (≥{MIN_DAYS_PER_YEAR} days)",
            })
            view["Year"] = view["Year"].astype(str)
            show_table(view, column_config={
                "Median price (₹/quintal)": st.column_config.NumberColumn(format="₹%.0f"),
                "Median premium (%)": st.column_config.NumberColumn(format="%.2f%%"),
                "Days above peers (%)": st.column_config.NumberColumn(format="%.1f%%"),
            })
            st.caption(
                f"Years with fewer than {MIN_DAYS_PER_YEAR} reporting days are shown "
                "but not used for the pattern label above."
            )

    with tab_data:
        table = comp[
            ["Arrival_Date", "Modal_Price", "Peer_Median_Price", "Premium_Discount_Pct", "Peer_Market_Count"]
        ].rename(columns={
            "Arrival_Date": "Date", "Modal_Price": "Selected market",
            "Peer_Median_Price": "Peer median", "Premium_Discount_Pct": "Premium (%)",
            "Peer_Market_Count": "Other reporting markets",
        }).sort_values("Date", ascending=False)
        show_table(table, column_config={
            "Date": st.column_config.DateColumn(format="DD MMM YYYY"),
            "Selected market": st.column_config.NumberColumn(format="₹%.2f"),
            "Peer median": st.column_config.NumberColumn(format="₹%.2f"),
            "Premium (%)": st.column_config.NumberColumn(format="%.2f%%"),
        })
        csv_button(table, "Download this table (CSV)", "market_history.csv", "exp_csv")


# ----------------------------------------------------------------------
# Page 2: market performance outlook
# ----------------------------------------------------------------------

def render_outlook(inference: pd.DataFrame) -> None:
    st.header("Market performance outlook")
    st.write(
        "Model estimates of how each market's modal price may compare with "
        "the peer median over the next seven calendar days. Ranks are national "
        "within each comparable group."
    )
    eligible = inference[inference["Eligible_For_Ranking"].fillna(False).astype(bool)]

    st.sidebar.subheader("Prediction filters")
    frame = eligible
    selections = {}
    for column, label in [("Commodity", "Commodity"), ("Variety", "Variety"), ("Grade", "Grade")]:
        options = option_list(frame, column)
        if not options:
            st.info("No ranked markets for this selection.")
            return
        choice = st.sidebar.selectbox(label, options, key=f"out_{column}")
        frame = frame[frame[column] == choice]
        selections[column] = choice
    if frame.empty:
        st.info("No ranked markets for this selection.")
        return

    dates = (
        frame.loc[frame["Comparison_Group_Size"] >= MIN_COMPARISON_GROUP, "Arrival_Date"]
        .drop_duplicates().sort_values(ascending=False).tolist()
    )
    if not dates:
        st.warning("No comparable market groups for this selection.")
        return
    selected_date = st.sidebar.selectbox(
        "Observation date", dates,
        format_func=lambda d: pd.Timestamp(d).strftime("%d %b %Y"), key="out_date",
    )
    comparison = frame[frame["Arrival_Date"] == selected_date].copy()

    st.sidebar.subheader("Location")
    states = option_list(comparison, "State")
    state = st.sidebar.selectbox("State", ["All states"] + states, key="out_state")
    if state != "All states":
        comparison = comparison[comparison["State"] == state]
    districts = option_list(comparison, "District")
    district = st.sidebar.selectbox("District", ["All districts"] + districts, key="out_district")
    if district != "All districts":
        comparison = comparison[comparison["District"] == district]

    st.sidebar.subheader("Screening")
    hide_warned = st.sidebar.checkbox("Hide markets with warnings", key="out_hide")
    only_positive = st.sidebar.checkbox("Only markets with a positive lower estimate", key="out_pos")

    comparison["Current_Premium_Pct"] = np.where(
        comparison["Peer_Median_Price"] > 0,
        (comparison["Modal_Price"] / comparison["Peer_Median_Price"] - 1) * 100,
        np.nan,
    )
    comparison["Warnings"] = warning_text(comparison)
    extreme = comparison["Current_Premium_Pct"].abs() > 100
    comparison.loc[extreme, "Warnings"] = (
        comparison.loc[extreme, "Warnings"]
        + np.where(comparison.loc[extreme, "Warnings"] != "", "; ", "")
        + "Over 100% from peers: check unit and grade"
    )
    if hide_warned:
        comparison = comparison[comparison["Warnings"] == ""]
    if only_positive:
        comparison = comparison[comparison["Lower_Premium_Discount_Pct"] > 0]
    if comparison.empty:
        st.info("No markets match these filters.")
        return

    comparison = comparison.sort_values(
        ["Predicted_Rank", "Predicted_Premium_Discount_Pct"], ascending=[True, False]
    )
    st.subheader(f"{selections['Commodity']} | {pd.Timestamp(selected_date).strftime('%d %B %Y')}")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Markets shown", f"{len(comparison):,}")
    k2.metric("Median predicted premium", pct(comparison["Predicted_Premium_Discount_Pct"].median()))
    k3.metric("Positive lower estimate", f"{(comparison['Lower_Premium_Discount_Pct'] > 0).sum():,}")
    k4.metric("With warnings", f"{(comparison['Warnings'] != '').sum():,}")

    tab_rank, tab_chart, tab_help = st.tabs(["Rankings", "Top 10 with uncertainty", "How to read this"])
    with tab_rank:
        table = comparison[
            ["Predicted_Rank", "Market", "District", "State", "Modal_Price",
             "Peer_Median_Price", "Current_Premium_Pct", "Predicted_Premium_Discount_Pct",
             "Lower_Premium_Discount_Pct", "Upper_Premium_Discount_Pct", "Peer_Market_Count", "Warnings"]
        ].rename(columns={
            "Predicted_Rank": "Rank", "Modal_Price": "Modal price",
            "Peer_Median_Price": "Peer median", "Current_Premium_Pct": "Current premium (%)",
            "Predicted_Premium_Discount_Pct": "Predicted premium (%)",
            "Lower_Premium_Discount_Pct": "Lower (Q10) %",
            "Upper_Premium_Discount_Pct": "Upper (Q90) %",
            "Peer_Market_Count": "Other markets",
        })
        show_table(table, column_config={
            "Modal price": st.column_config.NumberColumn(format="₹%.0f"),
            "Peer median": st.column_config.NumberColumn(format="₹%.0f"),
            "Current premium (%)": st.column_config.NumberColumn(format="%.2f%%"),
            "Predicted premium (%)": st.column_config.NumberColumn(format="%.2f%%"),
            "Lower (Q10) %": st.column_config.NumberColumn(format="%.2f%%"),
            "Upper (Q90) %": st.column_config.NumberColumn(format="%.2f%%"),
        })
        st.caption(
            "Filtering by state or district keeps the original national rank. Warnings flag records "
            "to double-check; they do not prove the data is wrong."
        )
        csv_button(table, "Download rankings (CSV)", "market_rankings.csv", "out_csv")

    with tab_chart:
        top = comparison.sort_values("Predicted_Rank").head(10).copy()
        top["Label"] = top["Market"].astype(str) + " (" + top["District"].astype(str) + ")"
        top["Label"] += np.where(
            top["Label"].duplicated(keep=False), " #" + top["Predicted_Rank"].astype(str), ""
        )
        top = top.rename(columns={
            "Predicted_Premium_Discount_Pct": "Predicted",
            "Lower_Premium_Discount_Pct": "Lower",
            "Upper_Premium_Discount_Pct": "Upper",
        })
        top["Lower estimate"] = np.where(
            top["Lower"] > 0, "Above peers even at Q10", "Range includes peer level"
        )
        order = top.sort_values("Predicted_Rank")["Label"].tolist()
        base = alt.Chart(top).encode(y=alt.Y("Label:N", sort=order, title=None))
        interval = base.mark_rule(strokeWidth=3, color=GREY).encode(
            x=alt.X("Lower:Q", title="Predicted premium vs peer median (%)"), x2="Upper:Q"
        )
        points = base.mark_point(filled=True, size=110).encode(
            x="Predicted:Q",
            color=alt.Color(
                "Lower estimate:N",
                scale=alt.Scale(
                    domain=["Above peers even at Q10", "Range includes peer level"],
                    range=[GREEN, "#d9a441"],
                ),
                legend=alt.Legend(orient="top", title=None),
            ),
            tooltip=[
                alt.Tooltip("Label:N", title="Market"),
                alt.Tooltip("Predicted:Q", format="+.2f", title="Predicted %"),
                alt.Tooltip("Lower:Q", format="+.2f", title="Q10 %"),
                alt.Tooltip("Upper:Q", format="+.2f", title="Q90 %"),
            ],
        )
        zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(
            strokeDash=[4, 4], color="#555"
        ).encode(x="x:Q")
        show_chart((interval + points + zero).properties(height=max(260, len(top) * 38)))
        st.caption(
            "Dot = median estimate, line = Q10 to Q90 range. A wide line means the model is unsure; "
            "prefer markets whose whole range sits above zero."
        )

    with tab_help:
        st.markdown("""
- **Predicted premium** is the expected gap between this market's price and the peer median over the
  next seven days. It is *not* a price rise from today.
- **Q10 / Q90** bound an empirical 80% range. Coverage is checked overall and by commodity but is not
  guaranteed for any single market.
- **Rank** is the national rank within markets reporting the same commodity, variety, grade and date.
- **Warnings** are screening flags: very high volatility, long reporting gaps, few peer markets, a very
  wide range, or a price more than 100% from peers. A giant premium is more often a unit or grade
  mismatch than a bargain.
        """)


# ----------------------------------------------------------------------
# Page 3: net realisation calculator
# ----------------------------------------------------------------------

def render_calculator(data_end: pd.Timestamp) -> None:
    st.header("Net realisation calculator")
    st.write(
        "Compare the estimated amount left after transport, fees and wastage. "
        "You control the expected selling price and cost assumptions; the historical "
        "modal price is only a reference point."
    )
    st.caption(
        "Choose the commodity, variety and grade in the sidebar before comparing markets. "
        "Use the same quality and quantity assumptions for a fair comparison."
    )

    st.sidebar.subheader("Produce")
    _, selections, complete = history_cascade(
        [("Commodity", "Commodity"), ("Variety", "Variety"), ("Grade", "Grade")],
        "calc",
    )
    if not complete:
        st.info("No records are available for this produce selection.")
        return

    latest = query_latest_prices(
        selections["Commodity"], selections["Variety"], selections["Grade"]
    )
    if latest.empty:
        st.info("No historical market prices were found for this selection.")
        return

    # ------------------------------------------------------------------
    # Step 1: quantity and unit
    # ------------------------------------------------------------------
    st.subheader("1. Set the quantity you plan to sell")
    unit_col, quantity_col = st.columns([1, 2])
    unit = unit_col.radio(
        "Quantity and price units",
        ["Quintal (100 kg)", "Kilogram"],
        horizontal=True,
        key="calc_unit",
    )
    per_kg = unit == "Kilogram"
    unit_word = "kg" if per_kg else "quintal"
    to_quintal = 0.01 if per_kg else 1.0
    price_to_qtl = 100.0 if per_kg else 1.0

    quantity = quantity_col.number_input(
        f"Quantity to sell ({unit_word})",
        min_value=0.01,
        value=1000.0 if per_kg else 10.0,
        step=50.0 if per_kg else 1.0,
        key=f"calc_qty_{unit_word}",
    )
    quintals = quantity * to_quintal

    st.divider()

    # ------------------------------------------------------------------
    # Step 2: markets
    # ------------------------------------------------------------------
    st.subheader("2. Choose markets to compare")
    states = ["All states"] + sorted(latest["State"].dropna().unique().tolist())
    state_filter = st.selectbox(
        "Narrow the market list by state", states, key="calc_state"
    )
    pool = latest if state_filter == "All states" else latest[latest["State"] == state_filter]
    available_labels = pool["Label"].tolist()
    if len(available_labels) < 2:
        st.warning("At least two markets with historical prices are needed for comparison.")
        return

    chosen = st.multiselect(
        "Select two to five markets",
        available_labels,
        max_selections=5,
        key="calc_markets",
        help="Choose markets you could realistically sell at, including your usual market.",
    )
    if len(chosen) < 2:
        st.info("Select at least two markets to continue.")
        return

    reference = st.selectbox(
        "Reference market for the comparison",
        chosen,
        key="calc_reference",
        help="Differences in the results table are measured against this market.",
    )

    st.divider()

    # ------------------------------------------------------------------
    # Step 3: price and cost assumptions
    # ------------------------------------------------------------------
    st.subheader("3. Enter expected prices and costs")
    st.caption(
        "Historical prices are pre-filled for convenience. Replace them with the prices "
        "you realistically expect to receive. Costs are entered as total rupees for the quantity above."
    )

    rows = []
    for market_index, label in enumerate(chosen):
        record = latest.loc[latest["Label"] == label].iloc[0]
        ref_price = (
            float(record["Modal_Price"])
            if pd.notna(record["Modal_Price"]) else 0.0
        )
        age_days = max(0, (data_end - record["Arrival_Date"]).days)
        key = f"{unit_word}|{label}"

        with st.expander(label, expanded=(market_index == 0)):
            st.caption(
                f"Latest historical modal price: {rupees(ref_price, 2)} per quintal "
                f"on {record['Arrival_Date'].strftime('%d %b %Y')}."
            )
            if age_days > STALE_PRICE_DAYS:
                st.warning(
                    f"This report is {age_days:,} days before the end of the historical dataset. "
                    "Treat it as an old reference, not a current quote."
                )

            price_col, transport_col, other_col = st.columns(3)
            price = price_col.number_input(
                f"Expected selling price (₹/{unit_word})",
                min_value=0.0,
                value=round(ref_price / price_to_qtl, 2),
                step=1.0 if per_kg else 50.0,
                key=f"price_{key}",
            )
            transport = transport_col.number_input(
                "Transport cost (₹ total)",
                min_value=0.0,
                value=0.0,
                step=100.0,
                key=f"transport_{key}",
            )
            other = other_col.number_input(
                "Loading, packing and other costs (₹)",
                min_value=0.0,
                value=0.0,
                step=100.0,
                key=f"other_{key}",
            )

            commission_col, wastage_col = st.columns(2)
            commission = commission_col.number_input(
                "Commission and fees (% of sales)",
                min_value=0.0,
                max_value=100.0,
                value=0.0,
                step=0.5,
                key=f"commission_{key}",
            )
            wastage = wastage_col.number_input(
                "Wastage or unsold quantity (%)",
                min_value=0.0,
                max_value=100.0,
                value=0.0,
                step=1.0,
                key=f"wastage_{key}",
            )

        # Convert the chosen quantity and unit price into a common quintal basis.
        sold_quintals = quintals * (1 - wastage / 100)
        gross = sold_quintals * price * price_to_qtl
        fees = gross * commission / 100
        net = gross - fees - transport - other
        rows.append({
            "Market": label,
            "Gross sales (₹)": gross,
            "Fees (₹)": fees,
            "Transport (₹)": transport,
            "Other costs (₹)": other,
            "Net (₹)": net,
            f"Net per {unit_word} (₹)": net / quantity,
            "_revenue_after_fees": gross - fees,
        })

    # ------------------------------------------------------------------
    # Step 4: compare the estimates
    # ------------------------------------------------------------------
    results = pd.DataFrame(rows)
    reference_rows = results.loc[results["Market"] == reference, "Net (₹)"]
    if reference_rows.empty:
        st.error("The reference market is no longer in the selected market list. Please select it again.")
        return

    reference_net = float(reference_rows.iloc[0])
    results["Net vs reference (₹)"] = results["Net (₹)"] - reference_net
    results = results.sort_values("Net (₹)", ascending=False).reset_index(drop=True)
    results.insert(0, "Rank", range(1, len(results) + 1))

    st.divider()
    st.subheader("4. Compare estimated net realisation")
    best = results.iloc[0]
    runner_up = results.iloc[1]
    lead = float(best["Net (₹)"] - runner_up["Net (₹)"])

    if best["Net (₹)"] <= 0:
        st.error("None of the selected markets produces a positive estimated net amount with these inputs.")
    else:
        st.success(f"Highest estimated net realisation: **{best['Market']}**")

    metric1, metric2, metric3 = st.columns(3)
    metric1.metric("Estimated net for this quantity", rupees(best["Net (₹)"], 2))
    metric2.metric(f"Net per {unit_word}", rupees(best[f"Net per {unit_word} (₹)"], 2))
    metric3.metric("Lead over the next market", rupees(lead, 2))

    if lead > 0 and best["_revenue_after_fees"] > 0:
        price_buffer = lead / best["_revenue_after_fees"] * 100
        st.info(
            f"The leading market's expected selling price could fall by about **{price_buffer:.1f}%** "
            "before its estimated net matches the second-place market, assuming all other inputs stay unchanged."
        )

    display_results = results.drop(columns="_revenue_after_fees")
    currency_columns = {
        col: st.column_config.NumberColumn(format="₹%.0f")
        for col in display_results.columns
        if col.endswith("(₹)") and not col.startswith("Net per")
    }
    currency_columns[f"Net per {unit_word} (₹)"] = st.column_config.NumberColumn(format="₹%.2f")
    show_table(display_results, column_config=currency_columns)

    chart = (
        alt.Chart(results)
        .mark_bar()
        .encode(
            y=alt.Y("Market:N", sort=results["Market"].tolist(), title=None),
            x=alt.X("Net (₹):Q", title="Estimated net realisation (₹)"),
            color=alt.condition(
                alt.datum["Net (₹)"] == float(best["Net (₹)"]),
                alt.value(GREEN),
                alt.value(GREY),
            ),
            tooltip=["Market:N", alt.Tooltip("Net (₹):Q", format=",.2f")],
        )
        .properties(height=max(180, len(results) * 48))
    )
    show_chart(chart)
    csv_button(
        display_results,
        "Download comparison (CSV)",
        "net_realisation.csv",
        "calc_csv",
    )
    st.caption(
        "Net = sales after wastage − commission and fees − transport − other costs. "
        "The estimate does not account for quality differences, buyer demand, bargaining, "
        "price changes during transport or any guarantee that a buyer will pay the entered price."
    )


# ----------------------------------------------------------------------
# Page 4: method and limits
# ----------------------------------------------------------------------

def render_method(data_end: pd.Timestamp, inference_end: pd.Timestamp) -> None:
    st.header("Method and limits")
    st.markdown(f"""
**Data.** Daily modal prices from Indian mandis (government Agmarknet reporting, redistributed as a
public dataset). The comparison data ends on **{data_end:%d %B %Y}**; the model rankings use
observations up to **{inference_end:%d %B %Y}**. Reporting volume dropped sharply after October
2025, and the cause has not been verified.

**Peer comparison.** For each market-day, the peer median is taken over other markets reporting the
same commodity, variety and grade that day (selected market excluded, at least three peers required).
Premium = market price / peer median - 1.

**Prediction.** LightGBM quantile models (Q10, Q50, Q90) estimate the median peer-relative log price
ratio over the following seven calendar days, with commodity-level interval calibration. Features use
only information available at the prediction date. Validation is chronological.

**Honest performance summary.** Against the strongest baseline ("the current relative position
persists"), the models reduced error by roughly 12% (MAE). The effect on which markets rank in the
top three was small. Treat rankings as a screening aid.

**What this is not.** Not a live price feed. Not a guarantee of price or profit. Not aware of transport
costs, quality, grade disputes or buyer demand. A very large premium usually signals a
data-comparability issue.
    """)


# ----------------------------------------------------------------------
# App shell
# ----------------------------------------------------------------------

STYLE = """
<style>
  .mm-title {
      font-size: 2.45rem;
      line-height: 1.15;
      font-weight: 750;
      color: #2f7d57;
      margin: 0;
  }
  .mm-sub {
      color: #64748b;
      font-size: 1.02rem;
      margin: 0.15rem 0 0.8rem 0;
  }
  div[data-testid="stMetric"] {
      background: rgba(47, 125, 87, 0.065);
      border: 1px solid rgba(47, 125, 87, 0.18);
      padding: 16px 16px;
      border-radius: 10px;
      min-height: 105px;
  }
  div[data-testid="stMetricLabel"] {
      color: #526477;
      font-weight: 600;
  }
  div[data-testid="stTabs"] button {
      font-weight: 600;
  }
</style>
"""


def main() -> None:
    st.set_page_config(
        page_title="MandiMitra | Agricultural Market Intelligence",
        page_icon="🌾",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(STYLE, unsafe_allow_html=True)

    missing = [path for path in (DATA_PATH, INFERENCE_PATH) if not path.exists()]
    if missing:
        st.error(
            "Required data files are missing:\n\n"
            + "\n".join(f"- `{path}`" for path in missing)
            + "\n\nCheck that the app data files are present in the repository."
        )
        st.stop()

    try:
        # Initialise the DuckDB view and load only the small inference table.
        get_history_connection()
        data_end = get_data_end()
        inference = load_inference()
        inference_end = pd.Timestamp(inference["Arrival_Date"].max())
    except Exception as exc:  # noqa: BLE001
        st.error("MandiMitra could not load its data. Check the deployment logs and data files.")
        st.caption(f"Details: {exc}")
        st.stop()

    st.markdown('<p class="mm-title">🌾 MandiMitra</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="mm-sub">Historical mandi price analysis and market comparison</p>',
        unsafe_allow_html=True,
    )
    st.write(
        "Explore historical prices, compare a mandi with its peers, review model-based "
        "market rankings and estimate net realisation after your own costs."
    )
    st.warning(
        f"Historical prices end on **{data_end:%d %b %Y}**. "
        f"The latest model observation is **{inference_end:%d %b %Y}**. "
        "This is not a live mandi-price feed, and its estimates are not guaranteed selling advice."
    )

    st.sidebar.title("🌾 MandiMitra")
    st.sidebar.caption(
        f"Historical data: {data_end:%d %b %Y}\n\n"
        f"Model observations: {inference_end:%d %b %Y}"
    )
    st.sidebar.divider()
    page = st.sidebar.radio(
        "Explore",
        [
            "Historical market explorer",
            "Market performance outlook",
            "Net realisation calculator",
        ],
        key="main_page",
    )

    if page == "Historical market explorer":
        render_explorer()
    elif page == "Market performance outlook":
        render_outlook(inference)
    else:
        render_calculator(data_end)

    st.divider()
    with st.expander("About the data, model and limitations", expanded=False):
        render_method(data_end, inference_end)

    st.divider()
    st.caption(
        "MandiMitra is an academic project. Historical prices and model estimates are for analysis only; "
        "they do not guarantee what any farmer will receive."
    )


if __name__ == "__main__":
    main()
