"""MandiMitra: historical agricultural market intelligence (Streamlit app).

The historical Parquet file is queried with DuckDB. It is never loaded
in full into Pandas, which keeps deployment memory use manageable.
"""

from __future__ import annotations

from pathlib import Path
import hashlib
import html
import os

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

# Warm paper + forest palette, with contrast-safe text colours.
PAPER = "#F6F1E7"
CARD = "#FFFDF8"
INK = "#1F2A24"
DEEP = "#123D2E"
GREEN = "#1F5D46"
LEAF = "#3E8E63"
GOLD = "#D49A1E"
GREY = "#758278"
RED = "#B74735"
MUTED = "#526257"
LINE = "#DED5C4"
FONT = "Manrope, 'Segoe UI', system-ui, sans-serif"
FRESH_OBSERVATION_DAYS = 30

# Keep Altair typography and axis labels readable against the warm chart canvas.
def _mandimitra_theme():
    return {
        "config": {
            "background": CARD,
            "view": {"stroke": "transparent"},
            "font": FONT,
            "axis": {
                "labelColor": INK, "titleColor": INK, "labelFont": FONT,
                "titleFont": FONT, "labelFontSize": 12, "titleFontSize": 12,
                "gridColor": "#EAE3D5", "domainColor": "#BEB5A5",
                "tickColor": "#BEB5A5", "titlePadding": 10,
            },
            "legend": {"labelColor": INK, "titleColor": INK, "labelFont": FONT,
                        "titleFont": FONT},
            "title": {"color": DEEP, "font": FONT},
            "text": {"color": INK, "font": FONT},
        }
    }

try:
    if hasattr(alt, "theme"):
        alt.theme.register("mandimitra", enable=True)(_mandimitra_theme)
    else:
        alt.themes.register("mandimitra", _mandimitra_theme)
        alt.themes.enable("mandimitra")
except Exception:
    # CSS still provides the application palette if an older Altair version
    # does not support the theme API used above.
    pass

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
    duckdb_threads = max(1, int(os.getenv("MANDIMITRA_DUCKDB_THREADS", "2")))
    duckdb_memory = os.getenv("MANDIMITRA_DUCKDB_MEMORY", "768MB")
    con.execute(f"SET threads = {duckdb_threads}")
    con.execute(f"SET memory_limit = '{duckdb_memory}'")
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


@st.cache_data(show_spinner="Loading model rankings...", max_entries=1)
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


@st.cache_data(show_spinner=False, max_entries=1)
def get_data_end() -> pd.Timestamp:
    rows = run_history_query(
        "SELECT MAX(Arrival_Date) AS data_end FROM historical_prices",
        as_dataframe=False,
    )
    if not rows or rows[0][0] is None:
        raise ValueError("The historical dataset contains no valid arrival dates.")
    return pd.Timestamp(rows[0][0])


@st.cache_data(show_spinner=False, max_entries=96)
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


@st.cache_data(show_spinner=False, max_entries=12)
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


@st.cache_data(show_spinner=False, max_entries=12)
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
    parts = []
    for column in present:
        values = frame[column]
        if pd.api.types.is_bool_dtype(values):
            mask = values.fillna(False)
        else:
            mask = values.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})
        parts.append(np.where(mask, FLAG_LABELS[column], ""))
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
        # When a parent filter changes, give the dependent widget a fresh key.
        # This prevents stale variety/district/market choices from carrying over.
        parent_signature = hashlib.sha1(repr(filters).encode("utf-8")).hexdigest()[:8]
        choice = st.sidebar.selectbox(
            label, options, key=f"{prefix}_{column}_{parent_signature}"
        )
        filters.append((column, choice))
        selections[column] = choice
    return filters, selections, True


def option_list(frame: pd.DataFrame, column: str) -> list:
    return sorted(frame[column].dropna().unique().tolist())


# ----------------------------------------------------------------------
# Branded presentation helpers
# ----------------------------------------------------------------------

esc = html.escape


def section(kicker: str, title: str, description: str = "") -> None:
    description_html = f'<p class="mm-section-desc">{esc(description)}</p>' if description else ""
    st.markdown(
        f'<div class="mm-section-kicker">{esc(kicker)}</div>'
        f'<h2 class="mm-section-title">{esc(title)}</h2>{description_html}',
        unsafe_allow_html=True,
    )


def callout(title: str, body: str, tone: str = "green") -> None:
    tone_class = tone if tone in {"green", "gold", "red"} else "green"
    st.markdown(
        f'<div class="mm-callout {tone_class}"><strong>{esc(title)}</strong> '
        f'{esc(body)}</div>',
        unsafe_allow_html=True,
    )


def render_hero(data_end: pd.Timestamp, inference_end: pd.Timestamp) -> None:
    st.markdown(
        '<div class="mm-hero"><div class="mm-hero-inner">'
        '<div class="mm-eyebrow">AGRICULTURAL MARKET INTELLIGENCE</div>'
        '<div class="mm-hero-title">MandiMitra</div>'
        '<p class="mm-hero-copy">Read the market before you move the produce. '
        'Explore historical mandi prices, compare peer performance and examine '
        'model-based outlooks, with uncertainty left in.</p>'
        f'<span class="mm-chip gold">Historical prices to {data_end:%d %b %Y}</span>'
        f'<span class="mm-chip">Model observations to {inference_end:%d %b %Y}</span>'
        '<span class="mm-chip">India-wide market coverage</span>'
        '</div><div class="mm-hero-orb"></div></div>',
        unsafe_allow_html=True,
    )


def render_podium_card(row: pd.Series, position: int) -> None:
    premium = float(row["Predicted_Premium_Discount_Pct"])
    colour_class = "positive" if premium >= 0 else "negative"
    market = esc(str(row["Market"]))
    district = esc(str(row["District"]))
    state = esc(str(row["State"]))
    lower = float(row["Lower_Premium_Discount_Pct"])
    upper = float(row["Upper_Premium_Discount_Pct"])
    st.markdown(
        f'<div class="mm-podium"><div class="mm-podium-rank">{position}</div>'
        f'<div class="mm-podium-market">{market}</div>'
        f'<div class="mm-podium-location">{district}, {state}</div>'
        f'<div class="mm-podium-value {colour_class}">{premium:+.1f}%</div>'
        f'<div class="mm-podium-range">Q10 {lower:+.1f}% · Q90 {upper:+.1f}%</div>'
        '</div>',
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------
# Page 1: historical market explorer
# ----------------------------------------------------------------------

def annual_summary(market_df: pd.DataFrame) -> pd.DataFrame:
    """Separate market reporting days from days with a valid peer comparison."""
    if market_df.empty:
        return pd.DataFrame()

    reports = market_df.copy()
    reports["Year"] = reports["Arrival_Date"].dt.year
    annual = reports.groupby("Year").agg(
        Reporting_Days=("Arrival_Date", "nunique"),
        Median_Price=("Modal_Price", "median"),
    )

    comparable = reports.dropna(subset=["Premium_Discount_Pct"]).copy()
    if comparable.empty:
        annual["Comparable_Days"] = 0
        annual["Median_Premium_Pct"] = np.nan
        annual["Days_Above_Peer_Pct"] = np.nan
    else:
        grouped = comparable.groupby("Year")
        comparisons = grouped.agg(
            Comparable_Days=("Arrival_Date", "nunique"),
            Median_Premium_Pct=("Premium_Discount_Pct", "median"),
        )
        comparisons["Days_Above_Peer_Pct"] = grouped["Premium_Discount_Pct"].apply(
            lambda values: (values > 0).mean() * 100
        )
        annual = annual.join(comparisons, how="left")
        annual["Comparable_Days"] = annual["Comparable_Days"].fillna(0).astype(int)

    # A consistency label is only meaningful when there are enough valid peer days.
    annual["Qualifies"] = annual["Comparable_Days"] >= MIN_DAYS_PER_YEAR
    return annual.reset_index()


def consistency_label(annual: pd.DataFrame) -> tuple[str, str]:
    if annual.empty:
        return "No data", "There are no market observations for this selection."
    qualifying = annual[annual["Qualifies"]]
    if len(qualifying) < MIN_QUALIFYING_YEARS:
        return (
            "Not enough history",
            f"Needs at least {MIN_QUALIFYING_YEARS} years with "
            f"{MIN_DAYS_PER_YEAR}+ comparable days; this market has {len(qualifying)}.",
        )
    share_positive = (qualifying["Median_Premium_Pct"] > 0).mean()
    median_premium = qualifying["Median_Premium_Pct"].median()
    if share_positive >= 0.75 and median_premium > 0:
        return "Consistently above peers", f"{share_positive:.0%} of qualifying years had a positive median premium."
    if share_positive <= 0.25 and median_premium < 0:
        return "Consistently below peers", f"Only {share_positive:.0%} of qualifying years had a positive median premium."
    return "Mixed", f"{share_positive:.0%} of qualifying years had a positive median premium."


def render_explorer() -> None:
    section(
        "HISTORICAL EXPLORER", "How one market compares with its peers",
        "Choose a commodity, variety, grade and mandi. All four analyses update together "
        "for the same market and date range.",
    )

    st.sidebar.subheader("Market selection")
    filters, sel, complete = history_cascade(
        [
            ("Commodity", "Commodity"), ("Variety", "Variety"),
            ("Grade", "Grade"), ("State", "State"),
            ("District", "District"), ("Market", "Market"),
        ],
        "exp",
    )
    if not complete:
        st.info("No records are available for this selection.")
        return

    market_df = query_market_history(tuple(filters))
    if market_df.empty:
        st.info("No records are available for this market.")
        return

    first, last = market_df["Arrival_Date"].min(), market_df["Arrival_Date"].max()
    if first < last:
        start_date, end_date = st.slider(
            "Historical date range", min_value=first.date(), max_value=last.date(),
            value=(first.date(), last.date()), format="DD MMM YYYY", key="exp_date_range",
        )
        market_df = market_df[
            (market_df["Arrival_Date"] >= pd.Timestamp(start_date))
            & (market_df["Arrival_Date"] < pd.Timestamp(end_date) + pd.Timedelta(days=1))
        ].copy()
    if market_df.empty:
        st.info("There are no reports in the selected date range.")
        return

    market_df = market_df.sort_values("Arrival_Date").reset_index(drop=True)
    comp = market_df.dropna(subset=["Premium_Discount_Pct", "Peer_Median_Price"]).copy()
    dates = market_df["Arrival_Date"].drop_duplicates().sort_values()
    gaps = dates.diff().dt.days.dropna()
    median_premium = comp["Premium_Discount_Pct"].median() if not comp.empty else np.nan
    days_above = (comp["Premium_Discount_Pct"] > 0).mean() * 100 if not comp.empty else np.nan

    st.markdown(
        f'<div class="mm-market-heading"><span>{esc(sel["Market"])}</span>'
        f'<small>{esc(sel["District"])}, {esc(sel["State"])} · '
        f'{esc(sel["Commodity"])} · {esc(sel["Variety"])} · {esc(sel["Grade"])}</small></div>',
        unsafe_allow_html=True,
    )

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Reporting days", f"{dates.nunique():,}", "all market reports")
    with k2:
        st.metric("Median modal price", rupees(market_df["Modal_Price"].median()), "per quintal")
    with k3:
        st.metric("Median premium vs peers", pct(median_premium), "comparable days only")
    with k4:
        st.metric("Days above peer median", pct(days_above, signed=False), f"of {len(comp):,} comparable days")

    annual = annual_summary(market_df)
    label, detail = consistency_label(annual)
    gap_text = (
        f" Typical reporting gap: {gaps.median():.0f} days; longest gap: {gaps.max():.0f} days."
        if len(gaps) else " There is only one reporting day in this selection."
    )
    callout(
        f"Historical pattern: {label}.",
        f"{detail}{gap_text} This describes the past, not future prices.",
        "gold" if label in {"Mixed", "Not enough history"} else "green",
    )

    # 1. Full-width price trend.
    section(
        "01 · PRICE TREND", "Modal price against the peer median",
        "The selected mandi is shown in forest green; the same-day peer median is shown in gold.",
    )
    price_rows = market_df.dropna(subset=["Modal_Price"]).copy()
    peer_rows = market_df.dropna(subset=["Peer_Median_Price"]).copy()
    x_axis = alt.X("Arrival_Date:T", title=None, axis=alt.Axis(labelOverlap=True))
    selected_line = alt.Chart(price_rows).mark_line(
        color=GREEN, strokeWidth=3.2, point=alt.OverlayMarkDef(size=38, filled=True),
    ).encode(
        x=x_axis,
        y=alt.Y("Modal_Price:Q", title="Modal price (₹ per quintal)", scale=alt.Scale(zero=False)),
        tooltip=[alt.Tooltip("Arrival_Date:T", title="Date"),
                 alt.Tooltip("Modal_Price:Q", title="Selected market (₹/quintal)", format=",.2f")],
    )
    peer_line = alt.Chart(peer_rows).mark_line(
        color=GOLD, strokeWidth=2.6, strokeDash=[7, 5],
    ).encode(
        x="Arrival_Date:T",
        y=alt.Y("Peer_Median_Price:Q", title="Modal price (₹ per quintal)", scale=alt.Scale(zero=False)),
        tooltip=[alt.Tooltip("Arrival_Date:T", title="Date"),
                 alt.Tooltip("Peer_Median_Price:Q", title="Peer median (₹/quintal)", format=",.2f")],
    )
    show_chart(alt.layer(selected_line, peer_line).properties(height=500).interactive(bind_y=False))
    st.caption(
        "Each point is an actual market report. Empty stretches are real reporting gaps, "
        "not interpolated values. The peer median excludes the selected market and requires "
        "at least three other reporting markets."
    )

    # 2. Full-width premium chart.
    section(
        "02 · PREMIUM OVER TIME", "Daily price difference from peers",
        "Premium = (selected market modal price ÷ peer median − 1) × 100. It is a price difference, not profit.",
    )
    if comp.empty:
        st.info("Peer comparisons are unavailable for the selected dates.")
    else:
        clip = st.checkbox("Clip the displayed axis at ±100%", value=True, key="exp_clip")
        premium_scale = alt.Scale(domain=[-100, 100], clamp=True) if clip else alt.Scale(zero=False)
        premium_plot = comp[["Arrival_Date", "Premium_Discount_Pct"]].copy()
        premium_plot["Direction"] = np.where(
            premium_plot["Premium_Discount_Pct"] >= 0, "Above peers", "Below peers"
        )
        bars = alt.Chart(premium_plot).mark_bar(size=5, opacity=0.88).encode(
            x=alt.X("Arrival_Date:T", title=None, axis=alt.Axis(labelOverlap=True)),
            y=alt.Y("Premium_Discount_Pct:Q", title="Premium vs peer median (%)", scale=premium_scale),
            color=alt.Color(
                "Direction:N",
                scale=alt.Scale(domain=["Above peers", "Below peers"], range=[LEAF, RED]),
                legend=alt.Legend(orient="top", title=None),
            ),
            tooltip=[alt.Tooltip("Arrival_Date:T", title="Date"),
                     alt.Tooltip("Premium_Discount_Pct:Q", title="Premium (%)", format="+.2f")],
        )
        zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=INK, strokeWidth=1.2).encode(y="y:Q")
        median_rule = alt.Chart(pd.DataFrame({"y": [float(median_premium)]})).mark_rule(
            color=GOLD, strokeDash=[7, 5], strokeWidth=2.3
        ).encode(y="y:Q")
        show_chart((bars + zero + median_rule).properties(height=440).interactive(bind_y=False))
        st.caption(
            "Gold dashed line = median premium for the selected period. If clipping is enabled, "
            "values outside the displayed range remain unchanged in the data and tooltips."
        )

    # 3. Full-width year-by-year chart and table.
    section(
        "03 · YEAR BY YEAR", "How the relationship changes across years",
        f"A year qualifies for the consistency label only when it has at least {MIN_DAYS_PER_YEAR} "
        "days with a valid peer comparison. Faded bars fall below that threshold.",
    )
    if annual.empty:
        st.info("There are no annual observations to summarise.")
    else:
        annual_chart = annual.copy()
        annual_chart["Year"] = annual_chart["Year"].astype(str)
        annual_chart["Premium label"] = annual_chart["Median_Premium_Pct"].map(
            lambda value: f"{value:+.1f}%" if pd.notna(value) else "n/a"
        )
        annual_bars = alt.Chart(annual_chart).mark_bar(
            cornerRadiusTopLeft=6, cornerRadiusTopRight=6, size=44,
        ).encode(
            x=alt.X("Year:N", title=None, axis=alt.Axis(labelAngle=0, labelFontSize=13)),
            y=alt.Y("Median_Premium_Pct:Q", title="Median premium (%)", scale=alt.Scale(zero=True)),
            color=alt.condition(
                alt.datum.Median_Premium_Pct >= 0, alt.value(LEAF), alt.value(RED)
            ),
            opacity=alt.condition(alt.datum.Qualifies, alt.value(1), alt.value(0.42)),
            tooltip=[alt.Tooltip("Year:N"),
                     alt.Tooltip("Reporting_Days:Q", title="Reporting days"),
                     alt.Tooltip("Comparable_Days:Q", title="Comparable days"),
                     alt.Tooltip("Median_Premium_Pct:Q", title="Median premium (%)", format="+.2f"),
                     alt.Tooltip("Days_Above_Peer_Pct:Q", title="Days above peers (%)", format=".1f")],
        )
        labels = alt.Chart(annual_chart).mark_text(
            dy=-12, fontWeight="bold", fontSize=13, color=INK,
        ).encode(x="Year:N", y="Median_Premium_Pct:Q", text="Premium label:N")
        zero_line = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=INK, opacity=0.65).encode(y="y:Q")
        show_chart((annual_bars + labels + zero_line).properties(height=380))

        annual_view = annual.rename(columns={
            "Reporting_Days": "Reporting days",
            "Comparable_Days": "Comparable days",
            "Median_Price": "Median modal price (₹/quintal)",
            "Median_Premium_Pct": "Median premium (%)",
            "Days_Above_Peer_Pct": "Days above peers (%)",
            "Qualifies": f"Qualifies (≥{MIN_DAYS_PER_YEAR} comparable days)",
        }).copy()
        annual_view["Year"] = annual_view["Year"].astype(str)
        st.markdown("**Year-by-year figures**")
        show_table(annual_view, height=280, column_config={
            "Median modal price (₹/quintal)": st.column_config.NumberColumn(format="₹%.0f"),
            "Median premium (%)": st.column_config.NumberColumn(format="%.2f%%"),
            "Days above peers (%)": st.column_config.NumberColumn(format="%.1f%%"),
        })

    # 4. Full-width daily table. Keep rows without a peer comparison visible.
    section(
        "04 · DAILY DATA", "Every reported day in the selected period",
        "Rows with no valid peer comparison are retained. Their peer median and premium will be blank.",
    )
    daily_table = market_df[
        ["Arrival_Date", "Modal_Price", "Peer_Median_Price", "Premium_Discount_Pct", "Peer_Market_Count"]
    ].rename(columns={
        "Arrival_Date": "Date",
        "Modal_Price": "Selected market (₹/quintal)",
        "Peer_Median_Price": "Peer median (₹/quintal)",
        "Premium_Discount_Pct": "Premium (%)",
        "Peer_Market_Count": "Other reporting markets",
    }).sort_values("Date", ascending=False)
    show_table(daily_table, height=460, column_config={
        "Date": st.column_config.DateColumn(format="DD MMM YYYY"),
        "Selected market (₹/quintal)": st.column_config.NumberColumn(format="₹%.2f"),
        "Peer median (₹/quintal)": st.column_config.NumberColumn(format="₹%.2f"),
        "Premium (%)": st.column_config.NumberColumn(format="%.2f%%"),
    })
    csv_button(daily_table, "Download daily data (CSV)", "market_history.csv", "exp_csv")


# ----------------------------------------------------------------------
# Page 2: market performance outlook
# ----------------------------------------------------------------------

def render_outlook(inference: pd.DataFrame) -> None:
    section(
        "MARKET OUTLOOK", "Seven-day relative performance",
        "Explore model estimates of how each mandi's modal price may compare with the peer median "
        "over the next seven calendar days. These are relative estimates, not price-rise forecasts.",
    )

    eligible_flag = inference["Eligible_For_Ranking"]
    if pd.api.types.is_bool_dtype(eligible_flag):
        eligible_mask = eligible_flag.fillna(False)
    else:
        eligible_mask = eligible_flag.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})
    eligible = inference[
        eligible_mask & (inference["Comparison_Group_Size"] >= MIN_COMPARISON_GROUP)
    ].copy()
    if eligible.empty:
        st.info("The model archive contains no eligible market rankings.")
        return

    st.sidebar.subheader("Prediction filters")
    include_old = st.sidebar.checkbox(
        f"Include observations older than {FRESH_OBSERVATION_DAYS} days",
        value=False, key="out_old",
        help="Markets are scored at their latest report. Older reports are hidden by default.",
    )
    inference_end = pd.Timestamp(eligible["Arrival_Date"].max())
    pool = eligible if include_old else eligible[
        eligible["Arrival_Date"] >= inference_end - pd.Timedelta(days=FRESH_OBSERVATION_DAYS)
    ]
    if pool.empty:
        st.info("No observations remain in the selected freshness window.")
        return

    frame = pool
    selections = {}
    for column, label in [("Commodity", "Commodity"), ("Variety", "Variety"), ("Grade", "Grade")]:
        options = option_list(frame, column)
        if not options:
            st.info("No ranked markets are available for this selection.")
            return
        upstream = "|".join(str(value) for value in selections.values())
        upstream_key = hashlib.sha1(upstream.encode("utf-8")).hexdigest()[:8]
        choice = st.sidebar.selectbox(label, options, key=f"out_{column}_{upstream_key}")
        frame = frame[frame[column] == choice]
        selections[column] = choice

    if frame.empty:
        st.info("No ranked markets are available for this selection.")
        return
    dates = frame["Arrival_Date"].drop_duplicates().sort_values(ascending=False).tolist()
    selection_key = hashlib.sha1("|".join(str(value) for value in selections.values()).encode("utf-8")).hexdigest()[:8]
    selected_date = st.sidebar.selectbox(
        "Observation date", dates,
        format_func=lambda value: pd.Timestamp(value).strftime("%d %b %Y"), key=f"out_date_{selection_key}",
    )
    comparison = frame[frame["Arrival_Date"] == selected_date].copy()

    st.sidebar.subheader("Location")
    states = option_list(comparison, "State")
    state = st.sidebar.selectbox("State", ["All states"] + states, key=f"out_state_{selection_key}_{pd.Timestamp(selected_date):%Y%m%d}")
    if state != "All states":
        comparison = comparison[comparison["State"] == state]
    districts = option_list(comparison, "District")
    district_scope = hashlib.sha1(f"{selection_key}|{selected_date}|{state}".encode("utf-8")).hexdigest()[:8]
    district = st.sidebar.selectbox("District", ["All districts"] + districts, key=f"out_district_{district_scope}")
    if district != "All districts":
        comparison = comparison[comparison["District"] == district]

    if comparison.empty:
        st.info("No markets are available for this location and date.")
        return

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

    # Report warning prevalence before the user applies screening toggles.
    warnings_before_screening = int((comparison["Warnings"] != "").sum())
    candidates_before_screening = len(comparison)
    hide_warned = st.sidebar.checkbox("Hide markets with warnings", key="out_hide")
    only_positive = st.sidebar.checkbox("Only markets with a positive lower estimate", key="out_pos")
    if hide_warned:
        comparison = comparison[comparison["Warnings"] == ""].copy()
    if only_positive:
        comparison = comparison[comparison["Lower_Premium_Discount_Pct"] > 0].copy()
    if comparison.empty:
        st.info("No markets match the screening filters. Adjust the filters in the sidebar.")
        return

    comparison = comparison.sort_values(
        ["Predicted_Rank", "Predicted_Premium_Discount_Pct"], ascending=[True, False]
    ).copy()
    comparison["Rank in view"] = (
        comparison["Predicted_Premium_Discount_Pct"].rank(method="min", ascending=False).astype(int)
    )

    st.markdown(
        f'<div class="mm-market-heading"><span>{esc(selections["Commodity"])}</span>'
        f'<small>{esc(selections["Variety"])} · {esc(selections["Grade"])} · '
        f'{pd.Timestamp(selected_date):%d %B %Y}</small></div>',
        unsafe_allow_html=True,
    )
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Markets shown", f"{len(comparison):,}", "after screening")
    with k2:
        st.metric("Median predicted premium", pct(comparison["Predicted_Premium_Discount_Pct"].median()), "versus peer median")
    with k3:
        st.metric("Positive lower estimate", f"{(comparison['Lower_Premium_Discount_Pct'] > 0).sum():,}", "Q10 remains above peers")
    with k4:
        st.metric("With warnings", f"{warnings_before_screening:,}", f"before screening · {candidates_before_screening:,} candidates")

    callout(
        "Read large premiums with care.",
        "A very large gap can reflect a unit, grade, location or reporting difference. "
        "The peer median is national within the matching commodity, variety and grade. "
        "Use these estimates as a screening aid, not selling advice.",
        "gold",
    )

    # Top three cards remain visible and use the model's original group ranks.
    section("LEADING MARKETS", "Highest-ranked opportunities")
    podium = comparison.sort_values("Predicted_Rank").head(3)
    cards = st.columns(max(1, len(podium)), gap="medium")
    for position, (column, (_, row)) in enumerate(zip(cards, podium.iterrows()), start=1):
        with column:
            render_podium_card(row, position)

    # Full-width uncertainty chart for legible market labels and ranges.
    section(
        "UNCERTAINTY", "Top 10 predicted markets with their 80% ranges",
        "The dot is the median estimate. The horizontal line runs from Q10 to Q90. "
        "A line entirely above zero has a positive lower estimate.",
    )
    top = comparison.sort_values("Predicted_Rank").head(10).copy()
    top["Label"] = top["Market"].astype(str) + " · " + top["District"].astype(str)
    top["Label"] += np.where(
        top["Label"].duplicated(keep=False), " #" + top["Predicted_Rank"].astype(int).astype(str), ""
    )
    top = top.rename(columns={
        "Predicted_Premium_Discount_Pct": "Predicted",
        "Lower_Premium_Discount_Pct": "Lower",
        "Upper_Premium_Discount_Pct": "Upper",
    })
    top["Signal"] = np.where(top["Lower"] > 0, "Positive even at Q10", "Range crosses peer level")
    order = top["Label"].tolist()
    base = alt.Chart(top).encode(y=alt.Y("Label:N", sort=order, title=None, axis=alt.Axis(labelFontSize=13)))
    intervals = base.mark_rule(strokeWidth=7, strokeCap="round", color="#BFB6A5").encode(
        x=alt.X("Lower:Q", title="Predicted premium vs peer median (%)"), x2="Upper:Q"
    )
    points = base.mark_point(filled=True, size=170, stroke=INK, strokeWidth=0.5).encode(
        x="Predicted:Q",
        color=alt.Color(
            "Signal:N",
            scale=alt.Scale(domain=["Positive even at Q10", "Range crosses peer level"], range=[GREEN, GOLD]),
            legend=alt.Legend(orient="top", title=None),
        ),
        tooltip=[alt.Tooltip("Label:N", title="Market"),
                 alt.Tooltip("Predicted:Q", title="Median estimate", format="+.2f"),
                 alt.Tooltip("Lower:Q", title="Q10", format="+.2f"),
                 alt.Tooltip("Upper:Q", title="Q90", format="+.2f")],
    )
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(
        strokeDash=[5, 5], color=INK, strokeWidth=1.5
    ).encode(x="x:Q")
    show_chart((intervals + points + zero).properties(height=max(470, len(top) * 48)))

    # Full-width current-versus-predicted relationship chart.
    section(
        "PERSISTENCE CHECK", "Current premium versus predicted premium",
        "Points close to the diagonal indicate that the prediction largely preserves today's relative position. "
        "The dashed line is the no-change reference.",
    )
    scatter_df = comparison.dropna(subset=["Current_Premium_Pct"]).copy()
    scatter_df["Screening status"] = np.where(scatter_df["Warnings"] != "", "Warning", "No warning")
    if len(scatter_df) < 3:
        st.info("Not enough markets are available to draw this comparison.")
    else:
        scatter_min = float(min(-100, scatter_df["Current_Premium_Pct"].min(), scatter_df["Predicted_Premium_Discount_Pct"].min()))
        scatter_max = float(max(100, scatter_df["Current_Premium_Pct"].max(), scatter_df["Predicted_Premium_Discount_Pct"].max()))
        padding = max(5, (scatter_max - scatter_min) * 0.04)
        domain = [scatter_min - padding, scatter_max + padding]
        diagonal = alt.Chart(pd.DataFrame({"x": domain, "y": domain})).mark_line(
            color=GREY, strokeDash=[5, 5], strokeWidth=2
        ).encode(x="x:Q", y="y:Q")
        dots = alt.Chart(scatter_df).mark_circle(size=95, opacity=0.8, stroke=CARD, strokeWidth=0.8).encode(
            x=alt.X("Current_Premium_Pct:Q", title="Current premium (%)", scale=alt.Scale(domain=domain)),
            y=alt.Y("Predicted_Premium_Discount_Pct:Q", title="Predicted premium (%)", scale=alt.Scale(domain=domain)),
            color=alt.Color(
                "Screening status:N",
                scale=alt.Scale(domain=["No warning", "Warning"], range=[LEAF, RED]),
                legend=alt.Legend(orient="top", title=None),
            ),
            tooltip=["Market:N", "District:N", "State:N",
                     alt.Tooltip("Current_Premium_Pct:Q", title="Current (%)", format="+.2f"),
                     alt.Tooltip("Predicted_Premium_Discount_Pct:Q", title="Predicted (%)", format="+.2f")],
        )
        show_chart((diagonal + dots).properties(height=440).interactive())

    # Full-width ranked table is always visible.
    section("FULL RANKING", f"Every eligible market in view ({len(comparison):,})")
    table = comparison[
        ["Predicted_Rank", "Rank in view", "Market", "District", "State", "Modal_Price",
         "Peer_Median_Price", "Current_Premium_Pct", "Predicted_Premium_Discount_Pct",
         "Lower_Premium_Discount_Pct", "Upper_Premium_Discount_Pct", "Peer_Market_Count", "Warnings"]
    ].rename(columns={
        "Predicted_Rank": "Rank (comparable group)", "Modal_Price": "Modal price",
        "Peer_Median_Price": "Peer median", "Current_Premium_Pct": "Current premium (%)",
        "Predicted_Premium_Discount_Pct": "Predicted premium (%)",
        "Lower_Premium_Discount_Pct": "Lower (Q10) %", "Upper_Premium_Discount_Pct": "Upper (Q90) %",
        "Peer_Market_Count": "Other markets",
    })
    show_table(table, height=500, column_config={
        "Modal price": st.column_config.NumberColumn(format="₹%.0f"),
        "Peer median": st.column_config.NumberColumn(format="₹%.0f"),
        "Current premium (%)": st.column_config.NumberColumn(format="%.2f%%"),
        "Predicted premium (%)": st.column_config.NumberColumn(format="%.2f%%"),
        "Lower (Q10) %": st.column_config.NumberColumn(format="%.2f%%"),
        "Upper (Q90) %": st.column_config.NumberColumn(format="%.2f%%"),
    })
    st.caption(
        "Rank (comparable group) is the original rank among markets scored on the same date and "
        "produce combination. Rank in view re-ranks only the rows remaining after your filters. "
        "The full table is scrollable; download it to inspect every row offline."
    )
    csv_button(table, "Download rankings (CSV)", "market_rankings.csv", "out_csv")

    # Explanations are visible, not hidden in an expander.
    explainer_cols = st.columns(3, gap="medium")
    explanations = [
        ("Predicted premium", "Expected relative gap against the peer median over the next seven calendar days. It is not a price rise from today."),
        ("Q10 to Q90", "An empirically calibrated 80% prediction interval. Coverage varies by commodity and is not guaranteed for an individual market."),
        ("Warnings and ranks", "Warnings are checks, not proof of bad data. Original ranks are within a comparable group; location filters do not change them."),
    ]
    for column, (title, body) in zip(explainer_cols, explanations):
        with column:
            st.markdown(
                f'<div class="mm-note-card"><b>{esc(title)}</b><p>{esc(body)}</p></div>',
                unsafe_allow_html=True,
            )


# ----------------------------------------------------------------------
# Page 3: net realisation calculator
# ----------------------------------------------------------------------

def render_calculator(data_end: pd.Timestamp) -> None:
    section(
        "NET REALISATION CALCULATOR", "What you actually take home",
        "Compare estimated net proceeds after transport, fees and wastage. You choose the selling "
        "prices and cost assumptions; historical mandi prices are starting references only.",
    )
    st.sidebar.subheader("Produce")
    _, selections, complete = history_cascade(
        [("Commodity", "Commodity"), ("Variety", "Variety"), ("Grade", "Grade")], "calc"
    )
    if not complete:
        st.info("No records are available for this produce selection.")
        return

    selection_signature = "|".join(str(selections[column]) for column in ("Commodity", "Variety", "Grade"))
    scope = hashlib.sha1(selection_signature.encode("utf-8")).hexdigest()[:10]

    latest = query_latest_prices(
        selections["Commodity"], selections["Variety"], selections["Grade"]
    )
    if latest.empty:
        st.info("No historical market prices were found for this selection.")
        return

    section("STEP 1", "Set the quantity you plan to sell")
    unit_col, quantity_col = st.columns([1, 1.6], gap="large")
    unit = unit_col.radio(
        "Quantity and price units", ["Quintal (100 kg)", "Kilogram"],
        horizontal=True, key=f"calc_unit_{scope}",
    )
    per_kg = unit == "Kilogram"
    unit_word = "kg" if per_kg else "quintal"
    to_quintal = 0.01 if per_kg else 1.0
    price_to_qtl = 100.0 if per_kg else 1.0
    quantity = quantity_col.number_input(
        f"Quantity to sell ({unit_word})", min_value=0.01,
        value=1000.0 if per_kg else 10.0,
        step=50.0 if per_kg else 1.0,
        key=f"calc_quantity_{scope}_{unit_word}",
    )
    quintals = quantity * to_quintal

    section("STEP 2", "Choose the markets you want to compare")
    states = ["All states"] + sorted(latest["State"].dropna().unique().tolist())
    state_filter = st.selectbox(
        "Narrow the market list by state", states, key=f"calc_state_{scope}"
    )
    pool = latest if state_filter == "All states" else latest[latest["State"] == state_filter]
    available_labels = pool["Label"].tolist()
    if len(available_labels) < 2:
        st.warning("At least two markets with historical prices are needed for comparison.")
        return

    state_scope = hashlib.sha1(state_filter.encode("utf-8")).hexdigest()[:8]
    chosen = st.multiselect(
        "Choose two to five markets", available_labels, max_selections=5,
        key=f"calc_markets_{scope}_{state_scope}",
        help="Include your usual market as a reference point.",
    )
    if len(chosen) < 2:
        st.info("Select at least two markets to calculate a comparison.")
        return
    chosen_scope = hashlib.sha1("|".join(chosen).encode("utf-8")).hexdigest()[:8]
    reference = st.selectbox(
        "Reference market (differences are measured against it)", chosen,
        key=f"calc_reference_{scope}_{state_scope}_{chosen_scope}",
    )

    section("STEP 3", "Enter expected prices and costs")
    st.caption(
        "Historical prices are pre-filled for convenience. Replace them with realistic expected prices. "
        "All transport and other costs are total rupees for the quantity entered above."
    )

    rows = []
    for market_index, label in enumerate(chosen, start=1):
        record = latest.loc[latest["Label"] == label].iloc[0]
        ref_price = float(record["Modal_Price"]) if pd.notna(record["Modal_Price"]) else 0.0
        age_days = max(0, (data_end - record["Arrival_Date"]).days)
        label_scope = hashlib.sha1(label.encode("utf-8")).hexdigest()[:10]
        widget_scope = f"{scope}_{state_scope}_{unit_word}_{label_scope}"

        with st.container(border=True):
            st.markdown(f'<div class="mm-input-heading">{market_index}. {esc(label)}</div>', unsafe_allow_html=True)
            st.caption(
                f"Latest historical modal price: {rupees(ref_price, 2)} per quintal "
                f"on {record['Arrival_Date']:%d %b %Y}."
            )
            if age_days > STALE_PRICE_DAYS:
                st.warning(
                    f"This report is {age_days:,} days before the end of the dataset. "
                    "Treat it as an old reference, not a current quote."
                )
            price_col, transport_col, other_col = st.columns(3, gap="medium")
            price = price_col.number_input(
                f"Expected selling price (₹/{unit_word})", min_value=0.0,
                value=round(ref_price / price_to_qtl, 2),
                step=1.0 if per_kg else 50.0, key=f"calc_price_{widget_scope}",
            )
            transport = transport_col.number_input(
                "Transport (₹ total)", min_value=0.0, value=0.0, step=100.0,
                key=f"calc_transport_{widget_scope}",
            )
            other = other_col.number_input(
                "Loading, packing and other costs (₹)", min_value=0.0, value=0.0, step=100.0,
                key=f"calc_other_{widget_scope}",
            )
            commission_col, wastage_col = st.columns(2, gap="medium")
            commission = commission_col.number_input(
                "Commission and fees (% of sales)", min_value=0.0, max_value=100.0,
                value=0.0, step=0.5, key=f"calc_commission_{widget_scope}",
            )
            wastage = wastage_col.number_input(
                "Wastage or unsold quantity (%)", min_value=0.0, max_value=100.0,
                value=0.0, step=1.0, key=f"calc_wastage_{widget_scope}",
            )

        potential = quintals * price * price_to_qtl
        lost_to_wastage = potential * wastage / 100
        gross = potential - lost_to_wastage
        fees = gross * commission / 100
        net = gross - fees - transport - other
        rows.append({
            "Market": label,
            "Potential sales (₹)": potential,
            "Wastage (₹)": lost_to_wastage,
            "Gross sales (₹)": gross,
            "Fees (₹)": fees,
            "Transport (₹)": transport,
            "Other costs (₹)": other,
            "Net (₹)": net,
            f"Net per {unit_word} (₹)": net / quantity,
            "_after_fees": gross - fees,
        })

    results = pd.DataFrame(rows)
    ref_rows = results.loc[results["Market"] == reference, "Net (₹)"]
    if ref_rows.empty:
        st.error("The reference market is no longer selected. Choose it again.")
        return
    reference_net = float(ref_rows.iloc[0])
    results["Net vs reference (₹)"] = results["Net (₹)"] - reference_net
    results = results.sort_values("Net (₹)", ascending=False).reset_index(drop=True)
    results.insert(0, "Rank", range(1, len(results) + 1))

    section("STEP 4", "Compare estimated net realisation")
    best, runner_up = results.iloc[0], results.iloc[1]
    lead = float(best["Net (₹)"] - runner_up["Net (₹)"])
    bad = float(best["Net (₹)"]) <= 0
    price_buffer_text = ""
    if lead > 0 and float(best["_after_fees"]) > 0:
        buffer = lead / float(best["_after_fees"]) * 100
        price_buffer_text = (
            f" Its expected sale price could fall by about {buffer:.1f}% before the next market "
            "catches up, with all other assumptions unchanged."
        )
    verdict_class = "bad" if bad else "good"
    verdict_label = "No selected market has a positive estimated net" if bad else "Highest estimated take-home"
    st.markdown(
        f'<div class="mm-verdict {verdict_class}"><div class="mm-verdict-label">{verdict_label}</div>'
        f'<div class="mm-verdict-market">{esc(str(best["Market"]))}</div>'
        f'<div class="mm-verdict-amount">{rupees(best["Net (₹)"], 0)}</div>'
        f'<div class="mm-verdict-foot">{rupees(best[f"Net per {unit_word} (₹)"], 2)} per {unit_word} · '
        f'{rupees(lead, 0)} ahead of the next market.{esc(price_buffer_text)}</div></div>',
        unsafe_allow_html=True,
    )

    k1, k2, k3 = st.columns(3)
    with k1:
        st.metric("Estimated net for this quantity", rupees(best["Net (₹)"], 2))
    with k2:
        st.metric(f"Net per {unit_word}", rupees(best[f"Net per {unit_word} (₹)"], 2))
    with k3:
        st.metric("Lead over the next market", rupees(lead, 2))

    display_results = results.drop(columns="_after_fees")
    currency_columns = {
        column: st.column_config.NumberColumn(format="₹%.0f")
        for column in display_results.columns
        if column.endswith("(₹)") and not column.startswith("Net per")
    }
    currency_columns[f"Net per {unit_word} (₹)"] = st.column_config.NumberColumn(format="₹%.2f")
    section("RESULTS TABLE", "Market-by-market cost and net breakdown")
    show_table(display_results, height=330, column_config=currency_columns)
    csv_button(display_results, "Download comparison (CSV)", "net_realisation.csv", "calc_csv")

    section("NET COMPARISON", "Estimated take-home by market")
    net_chart = alt.Chart(results).mark_bar(
        cornerRadiusTopRight=7, cornerRadiusBottomRight=7, size=35,
    ).encode(
        y=alt.Y("Market:N", sort=results["Market"].tolist(), title=None,
                axis=alt.Axis(labelFontSize=13, labelLimit=320)),
        x=alt.X("Net (₹):Q", title="Estimated net realisation (₹)"),
        color=alt.condition(
            alt.datum["Net (₹)"] == float(best["Net (₹)"]),
            alt.value(GREEN), alt.value("#B9C4B8"),
        ),
        tooltip=["Market:N", alt.Tooltip("Net (₹):Q", format=",.2f"),
                 alt.Tooltip("Net vs reference (₹):Q", format="+,.2f")],
    )
    show_chart(net_chart.properties(height=max(340, len(results) * 70)))

    section("COST BREAKDOWN", "Where the potential sales value goes")
    parts = results.assign(Net_for_chart=results["Net (₹)"].clip(lower=0))[[
        "Market", "Wastage (₹)", "Fees (₹)", "Transport (₹)", "Other costs (₹)", "Net_for_chart"
    ]].rename(columns={
        "Wastage (₹)": "Wastage", "Fees (₹)": "Fees", "Transport (₹)": "Transport",
        "Other costs (₹)": "Other costs", "Net_for_chart": "Net remaining",
    })
    long_parts = parts.melt("Market", var_name="Component", value_name="Rupees")
    component_order = ["Net remaining", "Wastage", "Fees", "Transport", "Other costs"]
    breakdown = alt.Chart(long_parts).mark_bar(size=35).encode(
        y=alt.Y("Market:N", sort=results["Market"].tolist(), title=None,
                axis=alt.Axis(labelFontSize=13, labelLimit=320)),
        x=alt.X("Rupees:Q", stack="zero", title="Rupees (₹)"),
        color=alt.Color(
            "Component:N", sort=component_order,
            scale=alt.Scale(
                domain=component_order,
                range=[GREEN, "#C6B79B", GOLD, RED, "#87958C"],
            ),
            legend=alt.Legend(orient="top", title=None),
        ),
        tooltip=["Market:N", "Component:N", alt.Tooltip("Rupees:Q", format=",.2f")],
    )
    show_chart(breakdown.properties(height=max(340, len(results) * 70)))
    st.caption(
        "A negative net amount is shown as zero in the cost-breakdown bar so the cost components remain readable. "
        "The results table above retains the actual negative value. The estimate excludes quality differences, "
        "buyer demand, bargaining and price changes during transport."
    )


# ----------------------------------------------------------------------
# Page 4: method and limits
# ----------------------------------------------------------------------

def render_method(data_end: pd.Timestamp, inference_end: pd.Timestamp) -> None:
    section("ABOUT THE MODEL", "Data, method and limitations")
    cols = st.columns(3, gap="large")
    method_cards = [
        (
            "Data and peer comparison",
            f"Daily mandi modal-price observations. Historical comparison data ends {data_end:%d %B %Y}. "
            "For each market-day, the peer median uses other markets reporting the same commodity, "
            "variety and grade on that date; the selected market is excluded and at least three peers are required."
        ),
        (
            "Prediction and validation",
            f"Model observations extend through {inference_end:%d %B %Y}. LightGBM quantile models estimate "
            "peer-relative performance over the following seven calendar days. Validation is chronological; "
            "reported intervals are estimates, not guarantees."
        ),
        (
            "Limits and responsible use",
            "This is not a live price feed. A premium can reflect geography, grade, units or reporting practice. "
            "The model does not know transport costs, buyer demand, quality disputes or bargaining power. "
            "The calculator is only as realistic as the inputs entered."
        ),
    ]
    for column, (title, body) in zip(cols, method_cards):
        with column:
            st.markdown(
                f'<div class="mm-note-card"><b>{esc(title)}</b><p>{esc(body)}</p></div>',
                unsafe_allow_html=True,
            )


# ----------------------------------------------------------------------
# App shell
# ----------------------------------------------------------------------

STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,700&family=Manrope:wght@400;500;600;700;800&display=swap');
:root {
  --ink: #1F2A24; --deep: #123D2E; --forest: #1F5D46; --leaf: #3E8E63;
  --gold: #D49A1E; --terra: #B74735; --paper: #F6F1E7; --card: #FFFDF8;
  --line: #DED5C4; --muted: #526257;
}
html, body, .stApp {font-family:'Manrope','Segoe UI',system-ui,sans-serif !important; color:var(--ink) !important;}
.stApp {background:radial-gradient(900px 420px at 92% -8%,rgba(212,154,30,.13),transparent 60%),
  radial-gradient(700px 380px at -5% 12%,rgba(62,142,99,.10),transparent 60%),var(--paper) !important;}
.block-container {padding-top:1.2rem !important;padding-bottom:3.2rem !important;max-width:1500px !important;}
#MainMenu, footer {visibility:hidden;}
.stApp h1,.stApp h2,.stApp h3,.stApp h4 {color:var(--deep) !important;font-family:'Fraunces',Georgia,serif !important;}
.stApp p,.stApp li,.stApp label,.stApp [data-testid="stWidgetLabel"],
.stApp [data-testid="stCaptionContainer"],.stApp small {color:var(--ink) !important;}
.stApp [data-testid="stCaptionContainer"] p {color:var(--muted) !important;font-size:.88rem !important;}
.stApp [data-testid="stSidebar"] {background:#F0E9DC !important;border-right:1px solid var(--line);}
.stApp [data-testid="stSidebar"] * {color:var(--ink);}
.stApp [data-testid="stSidebar"] h1,.stApp [data-testid="stSidebar"] h2,.stApp [data-testid="stSidebar"] h3 {color:var(--deep) !important;}
.stApp [data-testid="stMetric"] {background:var(--card);border:1px solid var(--line);border-radius:16px;
  padding:16px 18px;min-height:112px;box-shadow:0 10px 24px -22px rgba(18,61,46,.55);}
.stApp [data-testid="stMetricLabel"] {color:#46564B !important;font-weight:700 !important;}
.stApp [data-testid="stMetricValue"] {color:var(--deep) !important;font-family:'Fraunces',Georgia,serif !important;font-weight:700 !important;}
.stApp [data-testid="stMetricDelta"] {color:var(--muted) !important;}
.stApp [data-testid="stAlert"] {border-radius:12px;}
.stApp [data-testid="stAlert"] p,.stApp [data-testid="stAlert"] span {color:var(--ink) !important;}
/* -----------------------------------------------------------------
   Form controls: force a white surface and high-contrast dark text.
   Streamlit's BaseWeb widgets can inherit the browser/theme dark input
   surface, so style both the outer widget and its inner editable value.
   ----------------------------------------------------------------- */
.stApp [data-testid="stSelectbox"] [data-baseweb="select"] > div,
.stApp [data-testid="stMultiSelect"] [data-baseweb="select"] > div,
.stApp [data-testid="stDateInput"] [data-baseweb="input"] > div,
.stApp [data-testid="stTimeInput"] [data-baseweb="input"] > div,
.stApp [data-testid="stTextInput"] [data-baseweb="input"] > div,
.stApp [data-testid="stNumberInput"] [data-baseweb="input"] > div,
.stApp [data-testid="stTextArea"] [data-baseweb="textarea"] > div,
.stApp [data-baseweb="input"] > div,
.stApp [data-baseweb="textarea"] > div,
.stApp [data-baseweb="select"] > div {
  background:#FFFFFF !important;
  background-color:#FFFFFF !important;
  border-color:#C9D2C9 !important;
  color:#1F2A24 !important;
  box-shadow:none !important;
}
.stApp input,
.stApp textarea,
.stApp [role="combobox"],
.stApp [data-baseweb="select"] input,
.stApp [data-baseweb="input"] input,
.stApp [data-baseweb="textarea"] textarea {
  background:#FFFFFF !important;
  background-color:#FFFFFF !important;
  color:#1F2A24 !important;
  -webkit-text-fill-color:#1F2A24 !important;
  caret-color:#1F5D46 !important;
  opacity:1 !important;
}
.stApp [data-baseweb="select"] *,
.stApp [data-baseweb="input"] *,
.stApp [data-baseweb="textarea"] *,
.stApp [data-testid="stSelectbox"] *,
.stApp [data-testid="stMultiSelect"] *,
.stApp [data-testid="stDateInput"] *,
.stApp [data-testid="stTimeInput"] *,
.stApp [data-testid="stTextInput"] *,
.stApp [data-testid="stNumberInput"] *,
.stApp [data-testid="stTextArea"] * {
  color:#1F2A24 !important;
}
.stApp input::placeholder,
.stApp textarea::placeholder,
.stApp [data-baseweb="select"] input::placeholder {
  color:#68776C !important;
  -webkit-text-fill-color:#68776C !important;
  opacity:1 !important;
}
/* Dropdown menu and options must also stay white with readable text. */
[data-baseweb="popover"],
[data-baseweb="popover"] > div,
[data-baseweb="menu"],
[role="listbox"] {
  background:#FFFFFF !important;
  color:#1F2A24 !important;
}
[role="option"],
[data-baseweb="menu"] li,
[data-baseweb="menu"] [role="option"] {
  background:#FFFFFF !important;
  color:#1F2A24 !important;
}
[role="option"]:hover,
[role="option"][aria-selected="true"],
[data-baseweb="menu"] li:hover {
  background:#EAF2EB !important;
  color:#123D2E !important;
}
.stApp [data-baseweb="select"] svg {color:#526257 !important;fill:#526257 !important;}
.stApp [data-baseweb="select"]:focus-within > div,
.stApp [data-baseweb="input"]:focus-within > div,
.stApp [data-baseweb="textarea"]:focus-within > div {
  border-color:#1F5D46 !important;
  box-shadow:0 0 0 1px #1F5D46 !important;
}
.stApp [data-testid="stMetric"] * {color:#1F2A24 !important;}
.stApp [data-testid="stMetricLabel"] {color:#46564B !important;font-weight:700 !important;}
.stApp [data-testid="stMetricValue"],
.stApp [data-testid="stMetricValue"] * {color:#123D2E !important;}
.stApp [data-testid="stMetricDelta"],
.stApp [data-testid="stMetricDelta"] * {color:#526257 !important;}
.stApp .stButton>button,.stApp .stDownloadButton>button {background:var(--forest) !important;color:#FFFFFF !important;
  border:1px solid var(--forest) !important;border-radius:10px !important;font-weight:700 !important;}
.stApp .stButton>button:hover,.stApp .stDownloadButton>button:hover {background:var(--deep) !important;color:#FFFFFF !important;}
.stApp [data-testid="stDataFrame"] {border:1px solid var(--line);border-radius:13px;overflow:hidden;background:var(--card);}
.stApp [data-testid="stVegaLiteChart"] {background:var(--card);border:1px solid var(--line);border-radius:18px;
  padding:12px 14px;box-shadow:0 12px 30px -24px rgba(18,61,46,.45);}
.stApp hr {border-color:var(--line) !important;}
.mm-sidebar-brand {font-family:'Fraunces',Georgia,serif;font-size:1.55rem;font-weight:700;color:var(--deep);margin-bottom:.1rem;}
.mm-sidebar-meta {font-size:.78rem;line-height:1.55;color:var(--muted);}
.mm-hero {position:relative;overflow:hidden;border-radius:24px;padding:40px 46px 36px;color:#F6F1E7;
  background:linear-gradient(118deg,#0F3528 0%,#1F5D46 58%,#2F7D5B 100%);
  box-shadow:0 26px 50px -30px rgba(18,61,46,.7);margin-bottom:1rem;}
.mm-hero:before {content:"";position:absolute;inset:0;pointer-events:none;
  background:repeating-linear-gradient(115deg,rgba(255,255,255,.035) 0 2px,transparent 2px 15px);}
.mm-hero:after {content:"";position:absolute;right:-80px;top:-110px;width:360px;height:360px;border-radius:50%;
  background:radial-gradient(circle,rgba(224,165,38,.38),transparent 68%);}
.mm-hero-inner {position:relative;z-index:1;max-width:900px;}
.mm-eyebrow {font-size:.73rem;letter-spacing:.22em;text-transform:uppercase;font-weight:800;color:#F4C75E;}
.mm-hero-title {font-family:'Fraunces',Georgia,serif;font-weight:700;font-size:3.3rem;line-height:1.04;
  margin:.35rem 0 .5rem;color:#FFF9EA;letter-spacing:-.02em;}
.stApp .mm-hero-copy {font-size:1.06rem;line-height:1.6;color:#F1F5ED !important;margin:0 0 1rem;max-width:760px;}
.mm-chip {display:inline-block;padding:7px 14px;margin:0 8px 7px 0;border-radius:999px;font-size:.76rem;font-weight:700;
  background:rgba(255,255,255,.13);border:1px solid rgba(255,255,255,.28);color:#FFFFFF !important;}
.mm-chip.gold {background:rgba(224,165,38,.23);border-color:rgba(244,199,94,.65);color:#FFF1C5 !important;}
.mm-section-kicker {font-size:.72rem;letter-spacing:.2em;text-transform:uppercase;font-weight:800;color:#986A08;margin-top:1.65rem;}
.mm-section-title {font-family:'Fraunces',Georgia,serif !important;font-size:1.95rem;font-weight:700;color:var(--deep) !important;
  margin:.1rem 0 .25rem;line-height:1.2;}
.stApp .mm-section-desc {color:var(--muted) !important;font-size:.95rem;max-width:1000px;margin:.05rem 0 .8rem;}
.mm-market-heading {margin:1.1rem 0 .75rem;font-family:'Fraunces',Georgia,serif;font-size:1.8rem;font-weight:700;color:var(--deep);}
.mm-market-heading small {display:block;margin-top:.2rem;font-family:'Manrope',sans-serif;font-size:.86rem;font-weight:600;color:var(--muted) !important;}
.mm-callout {border-left:5px solid var(--forest);background:rgba(255,253,248,.92);border-radius:0 14px 14px 0;
  padding:14px 18px;margin:.9rem 0 1.1rem;color:var(--ink);box-shadow:0 8px 20px -18px rgba(18,61,46,.5);font-size:.94rem;line-height:1.55;}
.mm-callout strong {color:var(--deep);}
.mm-callout.gold {border-left-color:var(--gold);}.mm-callout.red {border-left-color:var(--terra);}
.mm-podium {position:relative;overflow:hidden;background:var(--card);border:1px solid var(--line);border-radius:19px;
  padding:20px 22px 18px;min-height:175px;box-shadow:0 18px 34px -26px rgba(18,61,46,.55);}
.mm-podium:first-child {border-color:#D8AC4D;}
.mm-podium-rank {position:absolute;right:14px;top:-9px;font-family:'Fraunces',Georgia,serif;font-size:5rem;font-weight:700;color:rgba(212,154,30,.22);}
.mm-podium-market {font-family:'Fraunces',Georgia,serif;font-size:1.25rem;font-weight:700;color:var(--deep);margin-right:38px;line-height:1.25;}
.mm-podium-location {font-size:.8rem;color:var(--muted) !important;margin:.25rem 0 .65rem;}
.mm-podium-value {font-family:'Fraunces',Georgia,serif;font-size:2.1rem;font-weight:700;line-height:1.1;}
.mm-podium-value.positive {color:var(--forest);}.mm-podium-value.negative {color:var(--terra);}
.mm-podium-range {font-size:.82rem;color:var(--muted) !important;margin-top:.25rem;}
.mm-note-card {height:100%;background:var(--card);border:1px solid var(--line);border-radius:15px;padding:17px 18px;
  box-shadow:0 10px 24px -22px rgba(18,61,46,.5);}
.mm-note-card b {display:block;font-family:'Fraunces',Georgia,serif;color:var(--deep);font-size:1.12rem;margin-bottom:.4rem;}
.mm-note-card p {color:var(--ink) !important;font-size:.88rem;line-height:1.55;margin:0;}
.mm-input-heading {font-family:'Fraunces',Georgia,serif;font-size:1.17rem;font-weight:700;color:var(--deep);margin-bottom:.2rem;}
.mm-verdict {border-radius:22px;padding:26px 30px;color:#FFF9EA;margin:.5rem 0 1rem;position:relative;overflow:hidden;
  background:linear-gradient(120deg,#0F3528,#1F5D46 70%,#2F7D5B);box-shadow:0 24px 44px -30px rgba(18,61,46,.8);}
.mm-verdict.bad {background:linear-gradient(120deg,#6E2B1C,#A8452F);}
.mm-verdict-label {font-size:.72rem;letter-spacing:.2em;text-transform:uppercase;font-weight:800;color:#FFE1A0;}
.mm-verdict-market {font-family:'Fraunces',Georgia,serif;font-weight:700;font-size:1.85rem;margin:.2rem 0 .1rem;line-height:1.2;color:#FFFFFF;}
.mm-verdict-amount {font-family:'Fraunces',Georgia,serif;font-weight:700;font-size:2.7rem;line-height:1.15;color:#FFFFFF;}
.mm-verdict-foot {color:#F1F5ED;font-size:.9rem;margin-top:.3rem;line-height:1.5;}
@media (max-width: 900px) {
  .mm-hero {padding:28px 24px;border-radius:18px;}
  .mm-hero-title {font-size:2.55rem;}
  .mm-section-title {font-size:1.6rem;}
  .block-container {padding-left:1rem !important;padding-right:1rem !important;}
}

/* -----------------------------------------------------------------
   Global readability pass: explicit light surfaces + dark text.
   These overrides intentionally come last so Streamlit/BaseWeb theme
   defaults cannot turn captions, metric subtitles or alerts black.
   ----------------------------------------------------------------- */
.stApp,
.stApp [data-testid="stAppViewContainer"],
.stApp [data-testid="stMain"],
.stApp [data-testid="stMainBlockContainer"] {
  color: #1F2A24 !important;
}

/* KPI / metric cards: label, value and the small explanatory line. */
.stApp [data-testid="stMetric"] {
  background: #FFFDF8 !important;
  background-color: #FFFDF8 !important;
  color: #1F2A24 !important;
  border: 1px solid #DED5C4 !important;
}
.stApp [data-testid="stMetric"] *,
.stApp [data-testid="stMetric"] p,
.stApp [data-testid="stMetric"] span,
.stApp [data-testid="stMetric"] label,
.stApp [data-testid="stMetric"] small,
.stApp [data-testid="stMetric"] div {
  background-color: transparent !important;
  -webkit-text-fill-color: currentColor !important;
  opacity: 1 !important;
}
.stApp [data-testid="stMetricLabel"],
.stApp [data-testid="stMetricLabel"] *,
.stApp [data-testid="stMetricLabel"] p {
  color: #46564B !important;
  -webkit-text-fill-color: #46564B !important;
}
.stApp [data-testid="stMetricValue"],
.stApp [data-testid="stMetricValue"] *,
.stApp [data-testid="stMetricValue"] p {
  color: #123D2E !important;
  -webkit-text-fill-color: #123D2E !important;
}
.stApp [data-testid="stMetricDelta"],
.stApp [data-testid="stMetricDelta"] *,
.stApp [data-testid="stMetricDelta"] p,
.stApp [data-testid="stMetricDelta"] span {
  color: #526257 !important;
  -webkit-text-fill-color: #526257 !important;
  background: transparent !important;
  background-color: transparent !important;
  opacity: 1 !important;
}

/* Captions and helper/subtitle text such as 'per quintal' and
   'comparable days only'. Keep them on a transparent light surface. */
.stApp [data-testid="stCaptionContainer"],
.stApp [data-testid="stCaptionContainer"] *,
.stApp [data-testid="stCaptionContainer"] p,
.stApp [data-testid="stHelp"],
.stApp [data-testid="stWidgetLabel"],
.stApp [data-testid="stWidgetLabel"] *,
.stApp .stMarkdown,
.stApp [data-testid="stMarkdownContainer"] {
  background: transparent !important;
  background-color: transparent !important;
}
.stApp [data-testid="stCaptionContainer"],
.stApp [data-testid="stCaptionContainer"] *,
.stApp [data-testid="stCaptionContainer"] p {
  color: #526257 !important;
  -webkit-text-fill-color: #526257 !important;
  opacity: 1 !important;
}
.stApp [data-testid="stMarkdownContainer"] p,
.stApp [data-testid="stMarkdownContainer"] li,
.stApp [data-testid="stMarkdownContainer"] span {
  color: #1F2A24;
}

/* Streamlit info, warning, success and error messages. Streamlit's
   active theme can otherwise give these notification boxes a dark fill. */
.stApp [data-testid="stAlert"],
.stApp [data-testid="stNotification"],
.stApp [data-baseweb="notification"] {
  background: #F1F5EF !important;
  background-color: #F1F5EF !important;
  color: #1F2A24 !important;
  border: 1px solid #D5E1D4 !important;
  border-radius: 12px !important;
  opacity: 1 !important;
}
.stApp [data-testid="stAlert"] > div,
.stApp [data-testid="stNotification"] > div,
.stApp [data-baseweb="notification"] > div {
  background: transparent !important;
  background-color: transparent !important;
}
.stApp [data-testid="stAlert"] *,
.stApp [data-testid="stNotification"] *,
.stApp [data-baseweb="notification"] * {
  color: #1F2A24 !important;
  -webkit-text-fill-color: #1F2A24 !important;
  background-color: transparent !important;
  opacity: 1 !important;
}
.stApp [data-testid="stAlert"] svg,
.stApp [data-testid="stNotification"] svg,
.stApp [data-baseweb="notification"] svg {
  fill: #1F5D46 !important;
  color: #1F5D46 !important;
}

/* All editable widgets, including number, text, date, time and text area. */
.stApp input,
.stApp textarea,
.stApp [role="combobox"],
.stApp [data-baseweb="input"] input,
.stApp [data-baseweb="textarea"] textarea,
.stApp [data-baseweb="select"] input {
  background: #FFFFFF !important;
  background-color: #FFFFFF !important;
  color: #1F2A24 !important;
  -webkit-text-fill-color: #1F2A24 !important;
  caret-color: #1F5D46 !important;
  opacity: 1 !important;
}
.stApp [data-testid="stTextInput"] [data-baseweb="input"],
.stApp [data-testid="stNumberInput"] [data-baseweb="input"],
.stApp [data-testid="stDateInput"] [data-baseweb="input"],
.stApp [data-testid="stTimeInput"] [data-baseweb="input"],
.stApp [data-testid="stTextArea"] [data-baseweb="textarea"],
.stApp [data-testid="stSelectbox"] [data-baseweb="select"],
.stApp [data-testid="stMultiSelect"] [data-baseweb="select"] {
  background: #FFFFFF !important;
  background-color: #FFFFFF !important;
  border-color: #C9D2C9 !important;
  color: #1F2A24 !important;
  opacity: 1 !important;
}
.stApp [data-testid="stNumberInput"] button {
  background: #F7F9F6 !important;
  color: #1F5D46 !important;
  border-color: #C9D2C9 !important;
}
.stApp [data-testid="stMultiSelect"] [data-baseweb="tag"] {
  background: #EAF2EB !important;
  color: #123D2E !important;
}
.stApp [data-testid="stMultiSelect"] [data-baseweb="tag"] *,
.stApp [data-testid="stMultiSelect"] [data-baseweb="tag"] span {
  color: #123D2E !important;
  -webkit-text-fill-color: #123D2E !important;
}
.stApp input::placeholder,
.stApp textarea::placeholder,
.stApp [data-baseweb="select"] input::placeholder {
  color: #68776C !important;
  -webkit-text-fill-color: #68776C !important;
  opacity: 1 !important;
}

/* Text around radio buttons, checkboxes, sliders and downloads. */
.stApp [data-testid="stCheckbox"] label,
.stApp [data-testid="stRadio"] label,
.stApp [data-testid="stSlider"] label,
.stApp [data-testid="stSelectSlider"] label,
.stApp [data-testid="stWidgetLabel"] p {
  color: #1F2A24 !important;
  -webkit-text-fill-color: #1F2A24 !important;
}
.stApp .stButton button,
.stApp .stDownloadButton button {
  color: #FFFFFF !important;
  -webkit-text-fill-color: #FFFFFF !important;
}

/* Keep custom cards and their supporting text readable too. */
.stApp .mm-callout,
.stApp .mm-callout * {
  color: #1F2A24 !important;
  -webkit-text-fill-color: #1F2A24 !important;
}
.stApp .mm-callout strong { color: #123D2E !important; -webkit-text-fill-color: #123D2E !important; }
.stApp .mm-note-card,
.stApp .mm-note-card p { color: #1F2A24 !important; -webkit-text-fill-color: #1F2A24 !important; }
.stApp .mm-section-desc,
.stApp .mm-market-heading small,
.stApp .mm-podium-location,
.stApp .mm-podium-range,
.stApp .mm-sidebar-meta {
  color: #526257 !important;
  -webkit-text-fill-color: #526257 !important;
}



/* Baby-pink accents for select dropdown arrow segments. The text/input
   portion stays white; only the narrow trailing icon area is pink. */
:root {
  --mm-baby-pink: #F7D2DE;
  --mm-baby-pink-border: #E8AFC1;
  --mm-baby-pink-deep: #A94F6C;
}
.stApp [data-testid="stSelectbox"] [data-baseweb="select"] > div > div:last-child,
.stApp [data-testid="stMultiSelect"] [data-baseweb="select"] > div > div:last-child {
  background: var(--mm-baby-pink) !important;
  background-color: var(--mm-baby-pink) !important;
  border-left: 1px solid var(--mm-baby-pink-border) !important;
  border-radius: 0 8px 8px 0 !important;
  color: var(--mm-baby-pink-deep) !important;
}
.stApp [data-testid="stSelectbox"] [data-baseweb="select"] > div > div:last-child svg,
.stApp [data-testid="stMultiSelect"] [data-baseweb="select"] > div > div:last-child svg,
.stApp [data-baseweb="select"] > div > div:last-child svg {
  color: var(--mm-baby-pink-deep) !important;
  fill: var(--mm-baby-pink-deep) !important;
  opacity: 1 !important;
}
/* Some Streamlit/BaseWeb versions put the arrow in an inner presentation
   wrapper rather than the final direct child. Cover that structure too. */
.stApp [data-testid="stSelectbox"] [data-baseweb="select"] [role="presentation"],
.stApp [data-testid="stMultiSelect"] [data-baseweb="select"] [role="presentation"] {
  background: var(--mm-baby-pink) !important;
  background-color: var(--mm-baby-pink) !important;
  color: var(--mm-baby-pink-deep) !important;
}

/* Baby-pink radio controls, including sidebar page navigation and radios
   used inside the calculator. Keep the option text dark and readable. */
.stApp [data-testid="stRadio"] input[type="radio"] {
  accent-color: #EFAFC2 !important;
}
.stApp [data-testid="stRadio"] [data-baseweb="radio"] > div:first-child {
  background: #FFFFFF !important;
  background-color: #FFFFFF !important;
  border: 2px solid var(--mm-baby-pink-border) !important;
  box-shadow: none !important;
}
.stApp [data-testid="stRadio"] [data-baseweb="radio"][aria-checked="true"] > div:first-child,
.stApp [data-testid="stRadio"] [data-baseweb="radio"][aria-checked="true"] > div:first-child {
  background: var(--mm-baby-pink) !important;
  background-color: var(--mm-baby-pink) !important;
  border-color: #D98DA6 !important;
}
.stApp [data-testid="stRadio"] [data-baseweb="radio"] [data-checked="true"],
.stApp [data-testid="stRadio"] [data-baseweb="radio"] svg {
  color: var(--mm-baby-pink-deep) !important;
  fill: var(--mm-baby-pink-deep) !important;
}
/* Fallback selectors for Streamlit versions that render radio circles
   with role=radio or nested label divs instead of data-baseweb=radio. */
.stApp [data-testid="stRadio"] [role="radio"] {
  accent-color: #EFAFC2 !important;
  border-color: var(--mm-baby-pink-border) !important;
}
.stApp [data-testid="stRadio"] [role="radio"][aria-checked="true"] {
  background-color: var(--mm-baby-pink) !important;
  border-color: #D98DA6 !important;
}
.stApp [data-testid="stRadio"] [role="radiogroup"] label > div:first-child > div:first-child {
  border-color: var(--mm-baby-pink-border) !important;
}
.stApp [data-testid="stRadio"] [role="radiogroup"] label[aria-checked="true"] > div:first-child > div:first-child,
.stApp [data-testid="stRadio"] [role="radiogroup"] label:has(input[type="radio"]:checked) > div:first-child > div:first-child {
  background: var(--mm-baby-pink) !important;
  background-color: var(--mm-baby-pink) !important;
  border-color: #D98DA6 !important;
}


/* ------------------------------------------------------------------
   Streamlit 1.59+ controls (React Aria DOM)
   The previous rules only covered the older BaseWeb select/radio DOM.
   Current Streamlit selectboxes use a real button with aria-label="Open",
   and radio indicators are StyledRadioOuter/StyledRadioInner elements.
   ------------------------------------------------------------------ */
:root {
  --mm-baby-pink: #F7D2DE;
  --mm-baby-pink-border: #E8AFC1;
  --mm-baby-pink-deep: #A94F6C;
  --mm-baby-pink-hover: #F1BFD0;
}

/* Selectbox: colour ONLY the narrow dropdown-arrow button.
   The text field to its left remains white. */
.stApp [data-testid="stSelectbox"] button[aria-label="Open"],
.stApp .stSelectbox button[aria-label="Open"],
.stApp [data-testid="stMultiSelect"] button[aria-label="Open"],
.stApp .stMultiSelect button[aria-label="Open"],
.stApp [data-testid="stMultiSelect"] button[aria-label="Show options"],
.stApp .stMultiSelect button[aria-label="Show options"] {
  background: var(--mm-baby-pink) !important;
  background-color: var(--mm-baby-pink) !important;
  color: var(--mm-baby-pink-deep) !important;
  -webkit-text-fill-color: var(--mm-baby-pink-deep) !important;
  border: 0 !important;
  border-left: 1px solid var(--mm-baby-pink-border) !important;
  border-radius: 0 8px 8px 0 !important;
  box-shadow: none !important;
  min-width: 2.45rem !important;
  align-self: stretch !important;
  opacity: 1 !important;
}

.stApp [data-testid="stSelectbox"] button[aria-label="Open"]:hover,
.stApp .stSelectbox button[aria-label="Open"]:hover,
.stApp [data-testid="stMultiSelect"] button[aria-label="Open"]:hover,
.stApp .stMultiSelect button[aria-label="Open"]:hover,
.stApp [data-testid="stMultiSelect"] button[aria-label="Show options"]:hover,
.stApp .stMultiSelect button[aria-label="Show options"]:hover {
  background: var(--mm-baby-pink-hover) !important;
  background-color: var(--mm-baby-pink-hover) !important;
}

.stApp [data-testid="stSelectbox"] button[aria-label="Open"] svg,
.stApp .stSelectbox button[aria-label="Open"] svg,
.stApp [data-testid="stMultiSelect"] button[aria-label="Open"] svg,
.stApp .stMultiSelect button[aria-label="Open"] svg,
.stApp [data-testid="stMultiSelect"] button[aria-label="Show options"] svg,
.stApp .stMultiSelect button[aria-label="Show options"] svg {
  color: var(--mm-baby-pink-deep) !important;
  fill: var(--mm-baby-pink-deep) !important;
  stroke: var(--mm-baby-pink-deep) !important;
  opacity: 1 !important;
}

.stApp [data-testid="stSelectbox"] button[aria-label="Open"]:focus-visible,
.stApp .stSelectbox button[aria-label="Open"]:focus-visible,
.stApp [data-testid="stMultiSelect"] button[aria-label="Open"]:focus-visible,
.stApp .stMultiSelect button[aria-label="Open"]:focus-visible {
  outline: 2px solid var(--mm-baby-pink-deep) !important;
  outline-offset: -2px !important;
}

/* Legacy BaseWeb fallback for deployments that have not migrated. */
.stApp [data-testid="stSelectbox"] [data-baseweb="select"] > div > div:last-child,
.stApp [data-testid="stMultiSelect"] [data-baseweb="select"] > div > div:last-child,
.stApp [data-testid="stSelectbox"] [data-baseweb="select"] [aria-hidden="true"] {
  background: var(--mm-baby-pink) !important;
  background-color: var(--mm-baby-pink) !important;
  border-left: 1px solid var(--mm-baby-pink-border) !important;
  color: var(--mm-baby-pink-deep) !important;
}

/* Radio buttons: the outer circle is a direct visual child of the radio
   option's row. Keep unselected circles white with a pink outline; fill the
   selected circle baby pink and preserve its small white centre. */
.stApp [data-testid="stRadioOption"] > div:first-of-type > div:first-child,
.stApp [data-testid="stRadioOption"] > div > div:first-child {
  background: #FFFFFF !important;
  background-color: #FFFFFF !important;
  border: 2px solid var(--mm-baby-pink-border) !important;
  box-shadow: none !important;
  transition: background-color 120ms ease, border-color 120ms ease !important;
}

.stApp [data-testid="stRadioOption"][data-selected="true"] > div:first-of-type > div:first-child,
.stApp [data-testid="stRadioOption"][aria-checked="true"] > div:first-of-type > div:first-child,
.stApp [data-testid="stRadioOption"]:has(input[type="radio"]:checked) > div:first-of-type > div:first-child,
.stApp [data-testid="stRadioOption"][data-selected="true"] > div > div:first-child,
.stApp [data-testid="stRadioOption"]:has(input[type="radio"]:checked) > div > div:first-child {
  background: var(--mm-baby-pink) !important;
  background-color: var(--mm-baby-pink) !important;
  border: 2px solid #D98DA6 !important;
}

.stApp [data-testid="stRadioOption"]:hover > div:first-of-type > div:first-child,
.stApp [data-testid="stRadioOption"]:hover > div > div:first-child {
  border-color: #D98DA6 !important;
}

/* Native-input fallback and the older BaseWeb radio implementation. */
.stApp [data-testid="stRadio"] input[type="radio"] {
  accent-color: var(--mm-baby-pink) !important;
}
.stApp [data-testid="stRadio"] label[data-baseweb="radio"] > div:first-child,
.stApp [data-testid="stRadio"] [role="radiogroup"] > label[data-baseweb="radio"] > div:first-child {
  background: #FFFFFF !important;
  border: 2px solid var(--mm-baby-pink-border) !important;
}
.stApp [data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) > div:first-child > div:first-child,
.stApp [data-testid="stRadio"] [role="radiogroup"] > label[data-baseweb="radio"]:has(input:checked) > div:first-child > div:first-child {
  background: var(--mm-baby-pink) !important;
  background-color: var(--mm-baby-pink) !important;
}

</style>
"""



def main() -> None:
    st.set_page_config(
        page_title="MandiMitra | Agricultural Market Intelligence",
        page_icon="🌾", layout="wide", initial_sidebar_state="expanded",
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
        get_history_connection()
        data_end = get_data_end()
        inference = load_inference()
        inference_end = pd.Timestamp(inference["Arrival_Date"].max())
    except Exception as exc:  # noqa: BLE001
        st.error("MandiMitra could not load its data. Check the deployment logs and data files.")
        st.caption(f"Details: {exc}")
        st.stop()

    st.sidebar.markdown(
        '<div class="mm-sidebar-brand">🌾 MandiMitra</div>'
        f'<div class="mm-sidebar-meta">Historical prices to {data_end:%d %b %Y}<br>'
        f'Model observations to {inference_end:%d %b %Y}</div>',
        unsafe_allow_html=True,
    )
    st.sidebar.divider()
    page = st.sidebar.radio(
        "Explore",
        ["Historical market explorer", "Market performance outlook", "Net realisation calculator"],
        key="main_page",
    )

    render_hero(data_end, inference_end)
    callout(
        "Historical data only.",
        f"Historical prices end on {data_end:%d %b %Y}. Model observations extend to "
        f"{inference_end:%d %b %Y}. This is not a live mandi-price feed, and estimates are not guaranteed selling advice.",
        "gold",
    )

    if page == "Historical market explorer":
        render_explorer()
    elif page == "Market performance outlook":
        render_outlook(inference)
    else:
        render_calculator(data_end)

    st.divider()
    render_method(data_end, inference_end)
    st.markdown(
        '<div style="text-align:center;color:#526257;font-size:.8rem;margin-top:2rem;line-height:1.6">'
        'MandiMitra is an academic project. Historical prices and model estimates are for analysis only; '
        'they do not guarantee what any farmer will receive.</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
