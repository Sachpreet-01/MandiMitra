
import altair as alt
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path


# ==========================================================
# 1. PROJECT PATHS
# ==========================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "app" / "market_comparison_app.parquet"

INFERENCE_PATH = (
    ROOT / "data" / "inference" /
    "mandimitra_historical_market_rankings.parquet"
)

MODEL_DIR = ROOT / "models"


# ==========================================================
# 2. PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="MandiMitra | Agricultural Market Intelligence",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================================
# 3. APPLICATION STYLING
# ==========================================================

st.markdown(
    """
    <style>
        .main-title {
            font-size: 2.5rem;
            font-weight: 750;
            color: #276749;
            margin-bottom: 0;
        }

        .subtitle {
            color: #64748b;
            font-size: 1rem;
            margin-top: 0;
        }

        .section-heading {
            color: #276749;
            font-weight: 650;
        }

        div[data-testid="stMetric"] {
            background-color: rgba(39, 103, 73, 0.06);
            border: 1px solid rgba(39, 103, 73, 0.15);
            padding: 14px;
            border-radius: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# ==========================================================
# 4. DATA VALIDATION
# ==========================================================

if not DATA_PATH.exists():
    st.error(f"Historical dataset not found:\n{DATA_PATH}")
    st.stop()

if not INFERENCE_PATH.exists():
    st.error(f"Inference dataset not found:\n{INFERENCE_PATH}")
    st.stop()


# ==========================================================
# 5. DATA LOADING
# ==========================================================

@st.cache_data
def load_data():
    data = pd.read_parquet(DATA_PATH)
    data["Arrival_Date"] = pd.to_datetime(data["Arrival_Date"])
    return data


@st.cache_data
def load_inference_data():
    predictions = pd.read_parquet(INFERENCE_PATH)
    predictions["Arrival_Date"] = pd.to_datetime(
        predictions["Arrival_Date"]
    )
    return predictions


try:
    df = load_data()
    inference_df = load_inference_data()

except Exception as e:
    st.error(f"Error loading datasets: {e}")
    st.stop()


# ==========================================================
# 6. APPLICATION HEADER
# ==========================================================

st.markdown(
    '<p class="main-title">🌾 MandiMitra</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">'
    'Agricultural Market Intelligence Platform'
    '</p>',
    unsafe_allow_html=True
)

st.write(
    "Explore agricultural market prices, compare historical "
    "market performance and examine predicted relative "
    "price opportunities."
)

st.warning(
    "This application currently uses historical data. "
    "The source dataset ends on 31 October 2025. "
    "Predictions are historical model outputs, not live "
    "market quotes or guaranteed selling advice."
)

st.divider()


# ==========================================================
# 7. SIDEBAR NAVIGATION
# ==========================================================

st.sidebar.title("🌾 MandiMitra")

page = st.sidebar.radio(
    "Navigation",
    [
        "Historical Market Explorer",
        "Market Performance Outlook",
        "Net Realisation Calculator"
    ]
)


# ==========================================================
# 8. HISTORICAL MARKET EXPLORER
# ==========================================================

if page == "Historical Market Explorer":

    st.header("Historical Market Explorer")

    st.write(
        "Explore historical modal prices and compare a "
        "selected market against other reporting markets "
        "for the same commodity, variety and grade."
    )

    # FILTERS

    st.sidebar.divider()
    st.sidebar.subheader("Historical Market Filters")

    commodities = sorted(df["Commodity"].dropna().unique())

    selected_commodity = st.sidebar.selectbox(
        "Select Vegetable",
        commodities
    )

    filtered_df = df[
        df["Commodity"] == selected_commodity
    ].copy()

    varieties = sorted(filtered_df["Variety"].dropna().unique())

    selected_variety = st.sidebar.selectbox(
        "Select Variety",
        varieties
    )

    filtered_df = filtered_df[
        filtered_df["Variety"] == selected_variety
    ]

    grades = sorted(filtered_df["Grade"].dropna().unique())

    selected_grade = st.sidebar.selectbox(
        "Select Grade",
        grades
    )

    filtered_df = filtered_df[
        filtered_df["Grade"] == selected_grade
    ]

    states = sorted(filtered_df["State"].dropna().unique())

    selected_state = st.sidebar.selectbox(
        "Select State",
        states
    )

    filtered_df = filtered_df[
        filtered_df["State"] == selected_state
    ]

    districts = sorted(filtered_df["District"].dropna().unique())

    selected_district = st.sidebar.selectbox(
        "Select District",
        districts
    )

    filtered_df = filtered_df[
        filtered_df["District"] == selected_district
    ]

    markets = sorted(filtered_df["Market"].dropna().unique())

    selected_market = st.sidebar.selectbox(
        "Select Market",
        markets
    )

    market_df = filtered_df[
        filtered_df["Market"] == selected_market
    ].copy()

    market_df = market_df.sort_values("Arrival_Date")

    # MARKET OVERVIEW

    st.subheader(
        f"{selected_market} | {selected_commodity}"
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Market Observations", f"{len(market_df):,}")

    col2.metric(
        "Median Modal Price",
        f"₹{market_df['Modal_Price'].median():,.0f}"
    )

    col3.metric(
        "Comparison Days",
        f"{market_df['Arrival_Date'].nunique():,}"
    )

    # HISTORICAL PRICE TREND

    st.divider()
    st.subheader("Historical Modal Price Trend")

    chart_df = (
        market_df
        .groupby("Arrival_Date")["Modal_Price"]
        .median()
        .reset_index()
        .sort_values("Arrival_Date")
    )

    if not chart_df.empty:
        st.line_chart(
            chart_df.set_index("Arrival_Date")["Modal_Price"],
            y_label="Modal Price (₹ per quintal)"
        )

    # PEER MARKET COMPARISON

    st.divider()
    st.subheader("Comparison with Other Markets")

    comparison_chart = market_df[
        [
            "Arrival_Date",
            "Modal_Price",
            "Peer_Median_Price",
            "Peer_Market_Count",
            "Premium_Discount_Pct"
        ]
    ].copy()

    comparison_chart = comparison_chart.rename(
        columns={
            "Modal_Price": "Selected Market",
            "Peer_Median_Price": "Peer Market Median",
            "Premium_Discount_Pct": "Price Premium (%)"
        }
    )

    comparison_chart = comparison_chart.sort_values("Arrival_Date")

    if not comparison_chart.empty:

        st.line_chart(
            comparison_chart.set_index("Arrival_Date")[
                ["Selected Market", "Peer Market Median"]
            ],
            y_label="Price (₹ per quintal)"
        )

        st.caption(
            "The peer median excludes the selected market. "
            "Each comparison uses at least three other "
            "reporting markets."
        )

        median_premium = comparison_chart[
            "Price Premium (%)"
        ].median()

        above_peer_pct = (
            comparison_chart["Price Premium (%)"]
            .gt(0)
            .mean() * 100
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

        # DAILY COMPARISON TABLE

        st.divider()
        st.subheader("Daily Market Comparison")

        display_df = comparison_chart[
            [
                "Arrival_Date",
                "Selected Market",
                "Peer Market Median",
                "Price Premium (%)",
                "Peer_Market_Count"
            ]
        ].copy()

        display_df = display_df.rename(
            columns={
                "Arrival_Date": "Date",
                "Peer_Market_Count": "Other Reporting Markets"
            }
        )

        st.dataframe(
            display_df.sort_values("Date"),
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
                ),
                "Other Reporting Markets": st.column_config.NumberColumn(
                    "Other Reporting Markets",
                    format="%d"
                )
            }
        )

    else:
        st.info(
            "No historical peer comparisons are available "
            "for this selection."
        )


# ==========================================================
# 9. MARKET PERFORMANCE OUTLOOK
# ==========================================================

elif page == "Market Performance Outlook":

    st.header("Market Performance Outlook")

    st.write(
        "Explore model predictions of how a market's price "
        "may compare with the median price of other reporting "
        "markets over the following seven calendar days."
    )

    st.info(
        "Predictions are based on the historical information "
        "available at each market's latest recorded observation. "
        "They do not represent live market forecasts."
    )

    # FILTERS

    st.sidebar.divider()
    st.sidebar.subheader("Prediction Filters")

    eligible_df = inference_df[
        inference_df["Eligible_For_Ranking"] == True
    ].copy()

    commodities = sorted(
        eligible_df["Commodity"].dropna().unique()
    )

    selected_commodity = st.sidebar.selectbox(
        "Select Vegetable",
        commodities,
        key="outlook_commodity"
    )

    filtered_df = eligible_df[
        eligible_df["Commodity"] == selected_commodity
    ].copy()

    varieties = sorted(filtered_df["Variety"].dropna().unique())

    selected_variety = st.sidebar.selectbox(
        "Select Variety",
        varieties,
        key="outlook_variety"
    )

    filtered_df = filtered_df[
        filtered_df["Variety"] == selected_variety
    ]

    grades = sorted(filtered_df["Grade"].dropna().unique())

    selected_grade = st.sidebar.selectbox(
        "Select Grade",
        grades,
        key="outlook_grade"
    )

    filtered_df = filtered_df[
        filtered_df["Grade"] == selected_grade
    ]

    # COMPARISON DATE

    valid_dates = sorted(
        filtered_df.loc[
            filtered_df["Comparison_Group_Size"] >= 5,
            "Arrival_Date"
        ].dropna().unique(),
        reverse=True
    )

    if not valid_dates:
        st.warning(
            "No comparable market groups are available "
            "for this selection."
        )
        st.stop()

    selected_date = st.sidebar.selectbox(
        "Select Observation Date",
        valid_dates,
        format_func=lambda x: pd.Timestamp(x).strftime(
            "%d %B %Y"
        ),
        key="outlook_date"
    )

    # NATIONAL RANKINGS ARE RETAINED BEFORE LOCATION FILTERS

    comparison_df = filtered_df[
        filtered_df["Arrival_Date"] == selected_date
    ].copy()

    comparison_df = comparison_df.sort_values("Predicted_Rank")

    # LOCATION FILTERS

    st.sidebar.subheader("Location Filters")

    states = sorted(comparison_df["State"].dropna().unique())

    selected_state = st.sidebar.selectbox(
        "State",
        ["All States"] + states,
        key="outlook_state"
    )

    if selected_state != "All States":
        comparison_df = comparison_df[
            comparison_df["State"] == selected_state
        ]

    districts = sorted(comparison_df["District"].dropna().unique())

    selected_district = st.sidebar.selectbox(
        "District",
        ["All Districts"] + districts,
        key="outlook_district"
    )

    if selected_district != "All Districts":
        comparison_df = comparison_df[
            comparison_df["District"] == selected_district
        ]

    # CURRENT PREMIUM CALCULATION

    comparison_df["Current_Premium_Pct"] = np.where(
        comparison_df["Peer_Median_Price"] > 0,
        (
            comparison_df["Modal_Price"]
            / comparison_df["Peer_Median_Price"] - 1
        ) * 100,
        np.nan
    )

    # OUTLOOK SUMMARY

    st.divider()

    st.subheader(
        f"{selected_commodity} | Market Outlook"
    )

    st.caption(
        f"Observation date: "
        f"{pd.Timestamp(selected_date).strftime('%d %B %Y')}"
    )

    if comparison_df.empty:
        st.info(
            "No eligible markets match the selected location."
        )
        st.stop()

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Markets Displayed",
        f"{len(comparison_df):,}"
    )

    col2.metric(
        "Median Predicted Premium",
        f"{comparison_df['Predicted_Premium_Discount_Pct'].median():+.2f}%"
    )

    positive_lower_bound_count = (
        comparison_df["Lower_Premium_Discount_Pct"] > 0
    ).sum()

    col3.metric(
        "Markets with Positive Lower Estimate",
        f"{positive_lower_bound_count:,}"
    )

    st.caption(
        "Market rankings are calculated against the full "
        "national comparison group before location filters "
        "are applied. Filtering by state or district does "
        "not change the original ranks."
    )

    # MARKET RANKINGS

    st.divider()
    st.subheader("Market Rankings")

    ranking_display = comparison_df[
        [
            "Predicted_Rank",
            "Market",
            "District",
            "State",
            "Modal_Price",
            "Peer_Median_Price",
            "Current_Premium_Pct",
            "Predicted_Premium_Discount_Pct",
            "Lower_Premium_Discount_Pct",
            "Upper_Premium_Discount_Pct",
            "Peer_Market_Count"
        ]
    ].copy()

    ranking_display = ranking_display.rename(
        columns={
            "Predicted_Rank": "Rank",
            "Modal_Price": "Current Modal Price",
            "Peer_Median_Price": "Current Peer Median",
            "Current_Premium_Pct": "Current Premium (%)",
            "Predicted_Premium_Discount_Pct": "Predicted Premium (%)",
            "Lower_Premium_Discount_Pct": "Lower Estimate (%)",
            "Upper_Premium_Discount_Pct": "Upper Estimate (%)",
            "Peer_Market_Count": "Other Reporting Markets"
        }
    )

    ranking_display = ranking_display.sort_values(
        ["Rank", "Predicted Premium (%)"],
        ascending=[True, False]
    )

    st.dataframe(
        ranking_display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Rank": st.column_config.NumberColumn(
                "Rank",
                format="%d"
            ),
            "Current Modal Price": st.column_config.NumberColumn(
                "Current Modal Price",
                format="₹%.2f"
            ),
            "Current Peer Median": st.column_config.NumberColumn(
                "Current Peer Median",
                format="₹%.2f"
            ),
            "Current Premium (%)": st.column_config.NumberColumn(
                "Current Premium (%)",
                format="%.2f%%"
            ),
            "Predicted Premium (%)": st.column_config.NumberColumn(
                "Predicted Premium (%)",
                format="%.2f%%"
            ),
            "Lower Estimate (%)": st.column_config.NumberColumn(
                "Lower Estimate (%)",
                format="%.2f%%"
            ),
            "Upper Estimate (%)": st.column_config.NumberColumn(
                "Upper Estimate (%)",
                format="%.2f%%"
            ),
            "Other Reporting Markets": st.column_config.NumberColumn(
                "Other Reporting Markets",
                format="%d"
            )
        }
    )

    # MARKET RISK INDICATORS

    st.divider()
    st.subheader("Market Risk Indicators")

    extreme_flag_count = 0

    if "Flag_Extreme_Current_Ratio" in comparison_df.columns:
        extreme_flag_count = int(
            comparison_df["Flag_Extreme_Current_Ratio"]
            .fillna(False)
            .sum()
        )

    high_premium_markets = comparison_df[
        comparison_df["Current_Premium_Pct"] > 100
    ].copy()

    high_premium_count = len(high_premium_markets)

    if extreme_flag_count > 0:
        st.warning(
            f"{extreme_flag_count} displayed market(s) have "
            "extreme current price-ratio quality flags."
        )

    if high_premium_count > 0:
        st.warning(
            f"{high_premium_count} market(s) currently have "
            "prices more than 100% above their peer median."
        )

        st.dataframe(
            high_premium_markets[
                [
                    "Market",
                    "State",
                    "Current_Premium_Pct",
                    "Modal_Price",
                    "Peer_Median_Price"
                ]
            ].rename(
                columns={
                    "Current_Premium_Pct": "Current Premium (%)",
                    "Modal_Price": "Current Modal Price",
                    "Peer_Median_Price": "Peer Median Price"
                }
            ).style.format({
                "Current Premium (%)": "{:.2f}%",
                "Current Modal Price": "₹{:,.0f}",
                "Peer Median Price": "₹{:,.0f}"
            }),
            use_container_width=True,
            hide_index=True
        )

    if extreme_flag_count == 0 and high_premium_count == 0:
        st.success(
            "No extreme price-ratio flags or premiums above "
            "100% were detected among the displayed markets."
        )

    st.caption(
        "A high current premium is a screening signal, not proof "
        "of incorrect data. Investigate product quality, reporting "
        "practices, market conditions and possible unit differences "
        "before interpreting it as a selling opportunity."
    )

    # TOP MARKET VISUALIZATION

    st.divider()
    st.subheader("Top 10 Predicted Markets")

    top10 = (
        comparison_df
        .sort_values("Predicted_Rank", ascending=True)
        .head(10)
        .sort_values(
            "Predicted_Premium_Discount_Pct",
            ascending=False
        )
    )

    if not top10.empty:

        chart_data = top10[
            [
                "Market",
                "Predicted_Premium_Discount_Pct"
            ]
        ].copy()

        market_order = chart_data["Market"].tolist()

        chart = (
            alt.Chart(chart_data)
            .mark_bar()
            .encode(
                x=alt.X(
                    "Predicted_Premium_Discount_Pct:Q",
                    title="Predicted Relative Premium (%)"
                ),
                y=alt.Y(
                    "Market:N",
                    sort=market_order,
                    title="Market"
                ),
                tooltip=[
                    alt.Tooltip("Market:N", title="Market"),
                    alt.Tooltip(
                        "Predicted_Premium_Discount_Pct:Q",
                        title="Predicted Premium",
                        format=".2f"
                    )
                ]
            )
            .properties(
                height=max(250, len(chart_data) * 35)
            )
        )

        st.altair_chart(
            chart,
            use_container_width=True
        )

    else:
        st.info("No market predictions are available to display.")

    # PREDICTION UNCERTAINTY

    st.divider()
    st.subheader("Understanding Prediction Uncertainty")

    st.write(
        "The model estimates three quantiles of future "
        "peer-relative performance:"
    )

    st.markdown(
        """
        - **Lower estimate (Q10):** Lower end of the model's
          predicted range.
        - **Median estimate (Q50):** Central prediction.
        - **Upper estimate (Q90):** Upper end of the model's
          predicted range.
        """
    )

    st.caption(
        "The interval is empirically calibrated using validation "
        "data. Its coverage is not guaranteed for every individual "
        "market or future period."
    )

    # DATA QUALITY WARNINGS

    st.divider()
    st.subheader("Data Quality and Reliability Warnings")

    warning_columns = {
        "Flag_Extreme_Current_Ratio": "Unusual current price relative to peers",
        "Flag_High_Volatility": "High recent price volatility",
        "Flag_Long_Reporting_Gap": "Long gap since previous market report",
        "Flag_Low_Peer_Coverage": "Limited peer market coverage",
        "Flag_Wide_Prediction_Interval": "Wide prediction interval"
    }

    warning_counts = {}

    for column, description in warning_columns.items():

        if column in comparison_df.columns:

            count = int(
                comparison_df[column].fillna(False).sum()
            )

            if count > 0:
                warning_counts[description] = count

    if warning_counts:

        for description, count in warning_counts.items():
            st.warning(f"{description}: {count} market(s)")

    else:

        st.success(
            "No listed quality warnings were detected "
            "for the displayed markets."
        )

    st.caption(
        "Quality flags are diagnostic warnings. They do not "
        "automatically establish that a market's reported price "
        "is incorrect."
    )


# ==========================================================
# 10. NET REALISATION CALCULATOR
# ==========================================================

elif page == "Net Realisation Calculator":

    st.header("Net Realisation Calculator")

    st.write(
        "Compare the estimated amount you would receive from "
        "selling your produce in different markets after "
        "deducting transportation and other selling costs."
    )

    st.info(
        "Enter your expected selling prices and costs manually. "
        "Historical market prices are available as reference "
        "information, but are not live quotes."
    )

    # COMMODITY AND PRODUCT FILTERS

    st.sidebar.divider()
    st.sidebar.subheader("Produce Details")

    commodities = sorted(df["Commodity"].dropna().unique())

    selected_commodity = st.sidebar.selectbox(
        "Select Vegetable",
        commodities,
        key="calculator_commodity"
    )

    calculator_df = df[
        df["Commodity"] == selected_commodity
    ].copy()

    varieties = sorted(
        calculator_df["Variety"].dropna().unique()
    )

    selected_variety = st.sidebar.selectbox(
        "Select Variety",
        varieties,
        key="calculator_variety"
    )

    calculator_df = calculator_df[
        calculator_df["Variety"] == selected_variety
    ]

    grades = sorted(
        calculator_df["Grade"].dropna().unique()
    )

    selected_grade = st.sidebar.selectbox(
        "Select Grade",
        grades,
        key="calculator_grade"
    )

    calculator_df = calculator_df[
        calculator_df["Grade"] == selected_grade
    ]

    # QUANTITY INPUT

    st.subheader("1. Enter Produce Quantity")

    col1, col2 = st.columns(2)

    with col1:
        quantity_kg = st.number_input(
            "Quantity to Sell (kg)",
            min_value=1.0,
            max_value=1000000.0,
            value=1000.0,
            step=50.0,
            key="calculator_quantity"
        )

    with col2:
        st.metric(
            "Quantity in Quintals",
            f"{quantity_kg / 100:,.2f}"
        )

    st.caption(
        "One quintal equals 100 kg. Enter the total quantity "
        "you expect to transport and sell."
    )

    # MARKET SELECTION

    st.divider()
    st.subheader("2. Select Markets to Compare")

    market_locations = (
        calculator_df[
            ["Market", "District", "State"]
        ]
        .drop_duplicates()
        .sort_values(["State", "District", "Market"])
        .copy()
    )

    market_locations["Market_Label"] = (
        market_locations["Market"].astype(str)
        + " | "
        + market_locations["District"].astype(str)
        + ", "
        + market_locations["State"].astype(str)
    )

    market_options = market_locations[
        "Market_Label"
    ].tolist()

    if len(market_options) < 2:

        st.warning(
            "At least two markets are required for comparison "
            "with this commodity, variety and grade."
        )
        st.stop()

    selected_market_labels = st.multiselect(
        "Choose up to five markets",
        options=market_options,
        default=market_options[:min(3, len(market_options))],
        max_selections=5,
        key="calculator_markets"
    )

    if len(selected_market_labels) < 2:

        st.warning(
            "Select at least two markets to calculate and "
            "compare net realisation."
        )
        st.stop()

    selected_locations = market_locations[
        market_locations["Market_Label"].isin(
            selected_market_labels
        )
    ].copy()

    # SELLING PRICE AND COST INPUTS

    st.divider()
    st.subheader("3. Enter Expected Prices and Costs")

    st.write(
        "Enter the expected selling price and estimated costs "
        "for each selected market. All monetary values are in INR."
    )

    calculation_inputs = []

    for index, (_, location) in enumerate(
        selected_locations.iterrows()
    ):

        market_name = location["Market"]
        district_name = location["District"]
        state_name = location["State"]

        market_label = (
            f"{market_name} | {district_name}, {state_name}"
        )

        with st.expander(
            market_label,
            expanded=True
        ):

            market_history = calculator_df[
                (calculator_df["Market"] == market_name)
                & (calculator_df["District"] == district_name)
                & (calculator_df["State"] == state_name)
            ].copy()

            market_history = market_history.dropna(
                subset=["Modal_Price", "Arrival_Date"]
            )

            if not market_history.empty:

                latest_record = market_history.sort_values(
                    "Arrival_Date"
                ).iloc[-1]

                reference_price = float(
                    latest_record["Modal_Price"]
                )

                reference_date = pd.Timestamp(
                    latest_record["Arrival_Date"]
                )

                st.caption(
                    f"Latest available historical modal price: "
                    f"₹{reference_price:,.2f} per quintal "
                    f"on {reference_date.strftime('%d %b %Y')}. "
                    "Use this only as a historical reference."
                )

                default_selling_price = reference_price / 100

            else:

                st.caption(
                    "No historical price is available for this "
                    "market selection. Enter the expected price manually."
                )

                default_selling_price = 0.0

            col1, col2 = st.columns(2)

            with col1:
                selling_price = st.number_input(
                    "Expected Selling Price (₹/kg)",
                    min_value=0.0,
                    value=float(default_selling_price),
                    step=0.5,
                    key=f"selling_price_{index}"
                )

                transport_cost = st.number_input(
                    "Transport Cost (₹)",
                    min_value=0.0,
                    value=0.0,
                    step=100.0,
                    key=f"transport_cost_{index}"
                )

            with col2:
                other_costs = st.number_input(
                    "Other Costs (₹)",
                    min_value=0.0,
                    value=0.0,
                    step=100.0,
                    key=f"other_costs_{index}"
                )

                st.caption(
                    "Other costs may include loading, unloading, "
                    "market fees, packaging or commissions."
                )

            calculation_inputs.append(
                {
                    "Market": market_name,
                    "District": district_name,
                    "State": state_name,
                    "Selling_Price_Per_Kg": selling_price,
                    "Transport_Cost": transport_cost,
                    "Other_Costs": other_costs
                }
            )

    # CALCULATE NET REALISATION

    st.divider()
    st.subheader("4. Net Realisation Results")

    results = []

    for item in calculation_inputs:

        gross_revenue = (
            quantity_kg * item["Selling_Price_Per_Kg"]
        )

        total_costs = (
            item["Transport_Cost"] + item["Other_Costs"]
        )

        net_realisation = gross_revenue - total_costs

        net_per_kg = net_realisation / quantity_kg

        results.append(
            {
                "Market": item["Market"],
                "District": item["District"],
                "State": item["State"],
                "Selling Price (₹/kg)": item["Selling_Price_Per_Kg"],
                "Gross Revenue (₹)": gross_revenue,
                "Transport Cost (₹)": item["Transport_Cost"],
                "Other Costs (₹)": item["Other_Costs"],
                "Total Costs (₹)": total_costs,
                "Net Realisation (₹)": net_realisation,
                "Net Realisation (₹/kg)": net_per_kg
            }
        )

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        "Net Realisation (₹)",
        ascending=False
    ).reset_index(drop=True)

    results_df.insert(
        0,
        "Comparison Rank",
        range(1, len(results_df) + 1)
    )

    best_market = results_df.iloc[0]

    # BEST RESULT

    st.success(
        f"Highest estimated net realisation: "
        f"{best_market['Market']}, "
        f"{best_market['District']}, "
        f"{best_market['State']}"
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Estimated Net Proceeds",
        f"₹{best_market['Net Realisation (₹)']:,.2f}"
    )

    col2.metric(
        "Net Realisation per kg",
        f"₹{best_market['Net Realisation (₹/kg)']:,.2f}"
    )

    if len(results_df) >= 2:

        difference = (
            results_df.iloc[0]["Net Realisation (₹)"]
            - results_df.iloc[1]["Net Realisation (₹)"]
        )

        col3.metric(
            "Advantage over Second Market",
            f"₹{difference:,.2f}"
        )

    # COMPARISON TABLE

    st.subheader("Market-wise Financial Comparison")

    st.dataframe(
        results_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Comparison Rank": st.column_config.NumberColumn(
                "Rank",
                format="%d"
            ),
            "Selling Price (₹/kg)": st.column_config.NumberColumn(
                "Selling Price (₹/kg)",
                format="₹%.2f"
            ),
            "Gross Revenue (₹)": st.column_config.NumberColumn(
                "Gross Revenue",
                format="₹%.2f"
            ),
            "Transport Cost (₹)": st.column_config.NumberColumn(
                "Transport Cost",
                format="₹%.2f"
            ),
            "Other Costs (₹)": st.column_config.NumberColumn(
                "Other Costs",
                format="₹%.2f"
            ),
            "Total Costs (₹)": st.column_config.NumberColumn(
                "Total Costs",
                format="₹%.2f"
            ),
            "Net Realisation (₹)": st.column_config.NumberColumn(
                "Net Realisation",
                format="₹%.2f"
            ),
            "Net Realisation (₹/kg)": st.column_config.NumberColumn(
                "Net Realisation (₹/kg)",
                format="₹%.2f"
            )
        }
    )

    # NET REALISATION CHART

    st.divider()
    st.subheader("Net Realisation by Market")

    chart_results = results_df[
        ["Market", "Net Realisation (₹)"]
    ].set_index("Market")

    st.bar_chart(
        chart_results,
        y_label="Estimated Net Realisation (₹)"
    )

    # FORMULA AND LIMITATIONS

    st.divider()
    st.subheader("How the Calculation Works")

    st.latex(
        r"\text{Gross Revenue} = "
        r"\text{Quantity (kg)} \times "
        r"\text{Selling Price per kg}"
    )

    st.latex(
        r"\text{Net Realisation} = "
        r"\text{Gross Revenue} - "
        r"\text{Transport Cost} - "
        r"\text{Other Costs}"
    )

    st.latex(
        r"\text{Net Realisation per kg} = "
        r"\frac{\text{Net Realisation}}{\text{Quantity (kg)}}"
    )

    st.caption(
        "This is an estimate based entirely on the values you enter. "
        "It does not account for unentered costs, differences in "
        "produce quality, spoilage, unsold quantities, price changes "
        "during transit or actual buyer demand. The highest calculated "
        "net realisation is not a guarantee of the highest actual profit."
    )


# ==========================================================
# 11. APPLICATION FOOTER
# ==========================================================

st.divider()

st.caption(
    "MandiMitra | Agricultural Market Intelligence | "
    "Developed as an academic and research project."
)

st.caption(
    "Historical price performance and model predictions "
    "do not guarantee actual selling prices or farmer profits. "
    "Transportation, quality, market fees and other costs "
    "must also be considered."
)
