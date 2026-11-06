
import streamlit as st
import pandas as pd
from pathlib import Path

# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "market_comparison.parquet"


# --------------------------------------------------
# 2. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="MandiMitra",
    page_icon="🌾",
    layout="wide"
)


# --------------------------------------------------
# 3. LOAD DATA
# --------------------------------------------------

@st.cache_data
def load_data():
    data = pd.read_parquet(DATA_PATH)
    data["Arrival_Date"] = pd.to_datetime(data["Arrival_Date"])
    return data


df = load_data()


# --------------------------------------------------
# 4. APPLICATION HEADER
# --------------------------------------------------

st.title("🌾 MandiMitra")
st.subheader("Agricultural Market Intelligence Platform")

st.caption(
    "Explore historical agricultural prices and compare markets across India."
)

st.warning(
    "Historical data only. Comparison period: January 2021 to October 2025. "
    "These results are not live market quotes or guaranteed selling advice."
)


# --------------------------------------------------
# 5. SIDEBAR FILTERS
# --------------------------------------------------

st.sidebar.header("Market Explorer")

commodities = sorted(df["Commodity"].dropna().unique())

selected_commodity = st.sidebar.selectbox(
    "Select Vegetable", commodities
)

filtered_df = df[
    df["Commodity"] == selected_commodity
].copy()

states = sorted(filtered_df["State"].dropna().unique())

selected_state = st.sidebar.selectbox(
    "Select State", states
)

filtered_df = filtered_df[
    filtered_df["State"] == selected_state
].copy()

districts = sorted(filtered_df["District"].dropna().unique())

selected_district = st.sidebar.selectbox(
    "Select District", districts
)

filtered_df = filtered_df[
    filtered_df["District"] == selected_district
].copy()

markets = sorted(filtered_df["Market"].dropna().unique())

selected_market = st.sidebar.selectbox(
    "Select Market", markets
)

market_df = filtered_df[
    filtered_df["Market"] == selected_market
].copy()

varieties = sorted(market_df["Variety"].dropna().unique())

selected_variety = st.sidebar.selectbox(
    "Select Variety", varieties
)

market_df = market_df[
    market_df["Variety"] == selected_variety
].copy()

grades = sorted(market_df["Grade"].dropna().unique())

selected_grade = st.sidebar.selectbox(
    "Select Grade", grades
)

market_df = market_df[
    market_df["Grade"] == selected_grade
].copy()


# --------------------------------------------------
# 6. MARKET OVERVIEW
# --------------------------------------------------

st.header(f"{selected_commodity} Market Explorer")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Market Observations",
    f"{len(market_df):,}"
)

col2.metric(
    "Median Modal Price",
    f"₹{market_df['Median_Modal_Price'].median():,.0f}"
)

col3.metric(
    "Comparison Days",
    f"{market_df['Arrival_Date'].nunique():,}"
)


# --------------------------------------------------
# 7. HISTORICAL PRICE TREND
# --------------------------------------------------

st.subheader("Historical Modal Price Trend")

chart_df = (
    market_df
    .groupby("Arrival_Date")["Median_Modal_Price"]
    .median()
    .reset_index()
    .sort_values("Arrival_Date")
)

if not chart_df.empty:
    st.line_chart(
        chart_df.set_index("Arrival_Date")["Median_Modal_Price"],
        y_label="Modal Price (₹ per quintal)"
    )



# --------------------------------------------------
# 8. PEER MARKET COMPARISON
# --------------------------------------------------

st.subheader("Comparison with Other Markets")

# Filter other markets for the same commodity, variety and grade
peer_rows = df[
    (df["Commodity"] == selected_commodity)
    & (df["Variety"] == selected_variety)
    & (df["Grade"] == selected_grade)
    & ~(
        (df["State"] == selected_state)
        & (df["District"] == selected_district)
        & (df["Market"] == selected_market)
    )
].copy()

# Calculate median price of peer markets for each arrival date
peer_df = (
    peer_rows
    .groupby("Arrival_Date")
    .agg(
        Peer_Median_Price=("Median_Modal_Price", "median")
    )
    .reset_index()
)

# Count distinct peer markets using complete market identity
peer_counts = (
    peer_rows
    .drop_duplicates(
        subset=[
            "Arrival_Date",
            "State",
            "District",
            "Market"
        ]
    )
    .groupby("Arrival_Date")
    .size()
    .reset_index(name="Peer_Market_Count")
)

# Merge peer median price with distinct market count
peer_df = peer_df.merge(
    peer_counts,
    on="Arrival_Date",
    how="inner"
)

# Require at least three distinct peer markets
peer_df = peer_df[
    peer_df["Peer_Market_Count"] >= 3
].copy()

# Merge selected market prices with peer market prices
comparison_chart = chart_df.merge(
    peer_df,
    on="Arrival_Date",
    how="inner"
).sort_values("Arrival_Date")

# Rename columns for chart readability
comparison_chart = comparison_chart.rename(
    columns={
        "Median_Modal_Price": "Selected Market",
        "Peer_Median_Price": "Peer Market Median"
    }
)

# --------------------------------------------------
# 9. PREMIUM INDICATOR
# --------------------------------------------------

if not comparison_chart.empty:

    st.line_chart(
        comparison_chart.set_index("Arrival_Date")[
            ["Selected Market", "Peer Market Median"]
        ],
        y_label="Price (₹ per quintal)"
    )

    st.caption(
        "Peer median excludes the selected market and requires at least "
        "three distinct other markets for each date."
    )

    comparison_chart["Premium_Pct"] = (
        (
            comparison_chart["Selected Market"]
            / comparison_chart["Peer Market Median"]
        ) - 1
    ) * 100

    median_premium = comparison_chart["Premium_Pct"].median()

    above_peer_pct = (
        comparison_chart["Premium_Pct"].gt(0).mean() * 100
    )

    evidence_days = len(comparison_chart)

    st.subheader("Market Price Premium Indicator")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Median Price Premium",
        f"{median_premium:+.2f}%"
    )

    col2.metric(
        "Days Above Peer Median",
        f"{above_peer_pct:.1f}%"
    )

    col3.metric(
        "Matched Comparison Days",
        f"{evidence_days:,}"
    )

# --------------------------------------------------
# 10. DETAILED COMPARISON TABLE
# --------------------------------------------------

st.subheader("Daily Market Comparison")

display_df = comparison_chart[
    [
        "Arrival_Date",
        "Selected Market",
        "Peer Market Median",
        "Premium_Pct",
        "Peer_Market_Count"
    ]
].copy()

display_df = display_df.rename(
    columns={
        "Arrival_Date": "Date",
        "Premium_Pct": "Price Premium (%)",
        "Peer_Market_Count": "Other Reporting Markets"
    }
)

# Ensure chronological sorting using actual dates
display_df = display_df.sort_values("Date")

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Date": st.column_config.DateColumn(
            "Date",
            format="DD MMM YYYY"
        ),
        "Selected Market": st.column_config.NumberColumn(
            "Selected Market",
            format="₹%.2f"
        ),
        "Peer Market Median": st.column_config.NumberColumn(
            "Peer Market Median",
            format="₹%.2f"
        ),
        "Price Premium (%)": st.column_config.NumberColumn(
            "Price Premium (%)",
            format="%.2f%%"
        )
    }
)

st.caption(
    "Historical price differences are not guaranteed future prices "
    "or estimates of farmer profit."
)
