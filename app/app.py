"""Streamlit interface for the Bank Customer Churn Prediction capstone."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from model_utils import load_assets, predict_customer


st.set_page_config(
    page_title="Bank Customer Churn Risk",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container {max-width: 1180px; padding-top: 2rem;}
      .hero {padding: 1.4rem 1.6rem; border-radius: 18px;
             background: linear-gradient(120deg, #082f49, #0f766e); color: white;
             margin-bottom: 1.25rem;}
      .hero h1 {margin: 0; font-size: 2.05rem; color: white;}
      .hero p {margin: .45rem 0 0; opacity: .9;}
      .result-card {border-radius: 16px; padding: 1rem 1.2rem;
                    border: 1px solid rgba(100,116,139,.28);}
      div[data-testid="stMetric"] {border: 1px solid rgba(100,116,139,.24);
                                   padding: .75rem; border-radius: 14px;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def cached_assets():
    return load_assets()


try:
    bundle, schema, metadata = cached_assets()
except Exception as error:
    st.error(f"The deployment package could not be loaded: {error}")
    st.stop()

APP_DIR = Path(__file__).resolve().parent
sample = pd.read_csv(APP_DIR / "sample_customer_input.csv").iloc[0].to_dict()

STATE_TO_ZONE = {
    "Benue": "North Central", "FCT": "North Central", "Kogi": "North Central",
    "Kwara": "North Central", "Nasarawa": "North Central", "Niger": "North Central",
    "Plateau": "North Central", "Adamawa": "North East", "Bauchi": "North East",
    "Borno": "North East", "Gombe": "North East", "Taraba": "North East",
    "Yobe": "North East", "Jigawa": "North West", "Kaduna": "North West",
    "Kano": "North West", "Katsina": "North West", "Kebbi": "North West",
    "Sokoto": "North West", "Zamfara": "North West", "Abia": "South East",
    "Anambra": "South East", "Ebonyi": "South East", "Enugu": "South East",
    "Imo": "South East", "Akwa Ibom": "South South", "Bayelsa": "South South",
    "Cross River": "South South", "Delta": "South South", "Edo": "South South",
    "Rivers": "South South", "Ekiti": "South West", "Lagos": "South West",
    "Ogun": "South West", "Ondo": "South West", "Osun": "South West",
    "Oyo": "South West",
}

LABELS = {
    "Monthly_Income_NGN": "Monthly income (NGN)",
    "Account_Balance_NGN": "Current account balance (NGN)",
    "Account_Balance_3M_Ago_NGN": "Account balance 3 months ago (NGN)",
    "Balance_Change_Pct_3M": "Balance change over 3 months (decimal)",
    "Digital_Transaction_Share": "Digital transaction share (0–1)",
    "Monthly_Transactions_3M_Ago": "Monthly transactions 3 months ago",
    "Monthly_Transactions_Current": "Current monthly transactions",
    "Transaction_Change_Pct_3M": "Transaction change over 3 months (decimal)",
    "Monthly_Transaction_Value_NGN": "Monthly transaction value (NGN)",
    "Months_Inactive_12M": "Inactive months in the last 12 months",
    "Days_Since_Last_Transaction": "Days since last transaction",
    "Failed_Transactions_3M": "Failed transactions in 3 months",
    "Customer_Service_Contacts_6M": "Customer-service contacts in 6 months",
    "Complaints_6M": "Complaints in 6 months",
    "Unresolved_Complaints_6M": "Unresolved complaints in 6 months",
    "Monthly_Bank_Fees_NGN": "Monthly bank fees (NGN)",
    "Fee_to_Income_Pct": "Fees as a percentage of income",
    "Loan_Amount_NGN": "Loan amount (NGN)",
    "USSD_Banking": "USSD banking",
}

HELP = {
    "Balance_Change_Pct_3M": "Use decimal form: -0.50 means a 50% decrease; 0.25 means a 25% increase.",
    "Digital_Transaction_Share": "Enter the share as a decimal between 0 and 1.",
    "Transaction_Change_Pct_3M": "Use decimal form: -1.00 means a 100% decrease; 0.50 means a 50% increase.",
    "Fee_to_Income_Pct": "This field is already a percentage. For example, 0.36 means 0.36%.",
}

GROUPS = {
    "Customer profile": [
        "Residence_Type", "Age", "Gender", "Marital_Status", "Education_Level",
        "Occupation", "Income_Regularity", "Monthly_Income_NGN",
    ],
    "Account relationship": [
        "Account_Type", "Salary_Account", "Tenure_Years", "Account_Balance_NGN",
        "Account_Balance_3M_Ago_NGN", "Balance_Change_Pct_3M", "Number_of_Products",
        "Has_Debit_Card", "Has_Credit_Card", "Monthly_Bank_Fees_NGN",
        "Fee_to_Income_Pct", "Active_Account_at_Another_Bank",
    ],
    "Channels and transactions": [
        "Mobile_Banking", "Internet_Banking", "USSD_Banking", "Preferred_Channel",
        "Digital_Transaction_Share", "Digital_Engagement",
        "Monthly_Transactions_3M_Ago", "Monthly_Transactions_Current",
        "Transaction_Change_Pct_3M", "Transaction_Frequency",
        "Monthly_Transaction_Value_NGN",
    ],
    "Engagement and service": [
        "Months_Inactive_12M", "Days_Since_Last_Transaction", "Active_Member",
        "Failed_Transactions_3M", "Customer_Service_Contacts_6M", "Complaints_6M",
        "Unresolved_Complaints_6M", "Average_Resolution_Days", "Satisfaction_Score",
    ],
    "Credit relationship": ["Loan_Status", "Loan_Amount_NGN", "Credit_Score"],
}


def readable_label(name: str) -> str:
    return LABELS.get(name, name.replace("_", " ").title())


def select_default(options: list[str], value: object) -> int:
    text = str(value)
    return options.index(text) if text in options else 0


def render_widget(name: str):
    categorical = schema["categorical_values_observed"]
    if name in categorical:
        options = categorical[name]
        return st.selectbox(
            readable_label(name),
            options,
            index=select_default(options, sample[name]),
            key=f"input_{name}",
        )

    limits = schema["numeric_ranges_observed"][name]
    dtype = schema["feature_dtypes"][name]
    minimum, maximum = limits["minimum_observed"], limits["maximum_observed"]
    value = sample[name]
    if dtype == "int64":
        return st.number_input(
            readable_label(name), min_value=int(minimum), max_value=int(maximum),
            value=int(value), step=1, help=HELP.get(name), key=f"input_{name}",
        )
    return st.number_input(
        readable_label(name), min_value=float(minimum), max_value=float(maximum),
        value=float(value), step=0.01, format="%.3f", help=HELP.get(name),
        key=f"input_{name}",
    )


st.markdown(
    """
    <div class="hero">
      <h1>Bank Customer Churn Risk</h1>
      <p>Decision-support prototype · TechCrush Cohort 7, Group 9</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.subheader("About this model")
    st.write(f"**Model:** {metadata['model_name']}")
    st.write(f"**Decision threshold:** {bundle['classification_threshold']:.2f}")
    st.write(f"**Required inputs:** {schema['number_of_features']}")
    st.info(
        "This educational proof of concept was trained entirely on synthetic data. "
        "It is not approved for real banking decisions."
    )
    with st.expander("How to interpret the result"):
        st.write(
            "The probability is the model's estimated churn risk within six months. "
            "A score at or above 0.57 is flagged for retention review."
        )

st.subheader("Customer information")
st.caption(
    "The fields are prefilled with the package's example customer. Change any value, "
    "then select **Assess churn risk**."
)

state_options = schema["categorical_values_observed"]["State"]
state = st.selectbox(
    "State", state_options, index=select_default(state_options, sample["State"]),
    key="input_State",
)
zone = STATE_TO_ZONE[state]
st.text_input("Geopolitical zone", value=zone, disabled=True)

values = {"State": state, "Geopolitical_Zone": zone}
tabs = st.tabs(list(GROUPS))
for tab, (_, fields) in zip(tabs, GROUPS.items()):
    with tab:
        left, right = st.columns(2)
        for index, field in enumerate(fields):
            with left if index % 2 == 0 else right:
                values[field] = render_widget(field)

st.divider()
button_col, note_col = st.columns([1, 2])
with button_col:
    assess = st.button("Assess churn risk", type="primary", use_container_width=True)
with note_col:
    st.caption("Predictions support review; they do not replace human judgement.")

if assess:
    try:
        st.session_state["prediction_result"] = predict_customer(values, bundle, schema)
    except Exception as error:
        st.error(f"Prediction failed: {error}")

if "prediction_result" in st.session_state:
    result = st.session_state["prediction_result"]
    probability = result["churn_probability"]
    churn_label = "Flagged for review" if result["predicted_class"] else "Not flagged"

    st.subheader("Assessment result")
    first, second, third = st.columns(3)
    first.metric("Churn probability", f"{probability:.1%}")
    second.metric("Model decision", churn_label)
    third.metric("Risk tier", result["risk_tier"])
    st.progress(probability, text=f"Predicted six-month churn risk: {probability:.1%}")

    recommendations = {
        "Low": "Continue routine service and periodic monitoring.",
        "Moderate": "Use lower-cost communication and monitor engagement changes.",
        "High": "Prioritise proactive contact and review possible service concerns.",
        "Very High": "Arrange prompt human review and a tailored retention conversation.",
    }
    if result["predicted_class"]:
        st.warning(recommendations[result["risk_tier"]])
    else:
        st.success(recommendations[result["risk_tier"]])

    with st.expander("Technical details"):
        st.write(f"Classification threshold: `{result['threshold']:.2f}`")
        st.write(f"Raw probability: `{probability:.12f}`")
        st.dataframe(result["input_frame"], use_container_width=True, hide_index=True)

st.divider()
st.caption(
    "Synthetic-data proof of concept. The model can produce false positives and false "
    "negatives and must not be used for automated consequential decisions."
)
