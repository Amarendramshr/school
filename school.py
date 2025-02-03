import pandas as pd
import streamlit as st
import os
import datetime
from prophet import Prophet
import numpy as np

DATA_FILE = "monitoring_data.csv"
SCHOOL_DATA_FILE = "district_schools.csv"  # CSV file containing districts and corresponding schools

# Load the CSV containing district and school names
district_school_data = pd.read_csv(SCHOOL_DATA_FILE)

st.markdown("<h1 style='text-align: center; color: #007bff;'>📊 Monitoring Cell Dashboard</h1>", unsafe_allow_html=True)

if "temp_data" not in st.session_state:
    st.session_state.temp_data = []

def reset_session_state():
    st.session_state.temp_data = []
    if os.path.exists(DATA_FILE):
        os.remove(DATA_FILE)
    st.success("Data reset successfully!")
st.button("Reset Data", on_click=reset_session_state)

st.sidebar.header("Data Entry")
col1, col2 = st.sidebar.columns(2)

# Initialize schools list in session state
if "schools" not in st.session_state:
    st.session_state.schools = []

with st.sidebar.form("data_form"):
    with col1:
        team_member = st.selectbox("👤 Team Member", ["Anand Mohan", "A Srivastava", "Sajan Snehi", "Sumi Sindhi", "A Raghuvanshi", "Shyam Mishra", "Jeet Kumar", "Shiv Pandit", "Biren Kumar"])
        district = st.selectbox("📍 District", district_school_data["District"].unique())

        # Callback to update school dropdown
        def update_schools():
            schools = district_school_data[district_school_data["District"] == district]["School"].tolist()
            st.session_state.schools = schools

        update_schools()

        # Update school dropdown based on selected district
        school_name = st.selectbox("🏫 School", st.session_state.schools)
        metric_name = st.selectbox("📊 Metric", ["Cleanliness", "Assembly activities", "Presence of Students", "Teachers' presence", "New Edu Init Imp", "Co-curricular Act.", "Others"])

    with col2:
        value = st.text_input("📈 Value", placeholder="Enter metric value (text or number)")
        is_anomaly = st.checkbox("Is this an anomaly?")
        anomaly_comment = st.text_area("💬 Anomaly Comment", placeholder="Enter comments for anomaly (if any)", height=70)

    add_metric = st.form_submit_button("✅ Add")

if add_metric:
    if not value:
        st.sidebar.error("⚠️ Please fill in the Value field.")
    else:
        try:
            float(value)
        except ValueError:
            pass  # If not convertible to float, treat as string
        new_entry = {
            "Team Member": team_member,
            "District": district,
            "School Name": school_name,
            "Metric Name": metric_name,
            "Value": value,
            "Is Anomaly": is_anomaly,
            "Anomaly Comment": anomaly_comment if is_anomaly else "",
            "Timestamp": datetime.date.today().strftime("%Y-%m-%d")
        }
        st.session_state.temp_data.append(new_entry)
        st.sidebar.success("✅ Metric added!")

        temp_df = pd.DataFrame(st.session_state.temp_data)
        if os.path.exists(DATA_FILE):
            existing_data = pd.read_csv(DATA_FILE)
            updated_data = pd.concat([existing_data, temp_df], ignore_index=True)
        else:
            updated_data = temp_df
        updated_data.to_csv(DATA_FILE, index=False)
        st.session_state.temp_data = []  # Clear after saving

if os.path.exists(DATA_FILE) and os.stat(DATA_FILE).st_size > 0:
    data = pd.read_csv(DATA_FILE)
    if "Is Anomaly" not in data.columns:  # Essential check!
        data["Is Anomaly"] = False
else:
    data = pd.DataFrame(columns=["Team Member", "District", "School Name", "Metric Name", "Value", "Is Anomaly", "Anomaly Comment", "Timestamp"])
    data["Is Anomaly"] = False  # Initialize even for a new DataFrame

# Interactive Filters
st.header("Interactive Filters")

# Filter by District
selected_district = st.multiselect("Filter by District", data["District"].unique())

# Filter by School
selected_school = st.multiselect("Filter by School", data["School Name"].unique())

# Filter by Metric
selected_metric = st.multiselect("Filter by Metric", data["Metric Name"].unique())

# Filter by Date Range
date_range = st.date_input("Filter by Date Range", [datetime.date.today() - datetime.timedelta(days=7), datetime.date.today()])

# Apply Filters
filtered_data = data[
    (data["District"].isin(selected_district)) &
    (data["School Name"].isin(selected_school)) &
    (data["Metric Name"].isin(selected_metric)) &
    (pd.to_datetime(data["Timestamp"]).dt.date >= date_range[0]) &
    (pd.to_datetime(data["Timestamp"]).dt.date <= date_range[1])
]

st.subheader("Filtered Data")
st.dataframe(filtered_data)

# Export Filtered Data
if st.button("Export Filtered Data as CSV"):
    st.download_button("⬇️ Download Filtered CSV", filtered_data.to_csv(index=False).encode('utf-8'), file_name="filtered_monitoring_data.csv", mime="text/csv")

# Anomaly-Based Trend Analysis
st.subheader("Anomaly-Based Trend Analysis")

analyze_trends = st.button("Analyze Trends based on Anomalies")

if analyze_trends:
    anomalies_data = filtered_data[filtered_data["Is Anomaly"] == True]

    if not anomalies_data.empty:
        # Convert 'Value' to numeric for trend analysis
        anomalies_data["Value"] = pd.to_numeric(anomalies_data["Value"], errors="coerce")
        anomalies_data.dropna(subset=["Value"], inplace=True)

        grouped_anomalies = anomalies_data.groupby(["School Name", "Metric Name"])

        for (school, metric), group in grouped_anomalies:
            st.write(f"Trend Analysis based on Anomalies for {school} - {metric}:")

            df_prophet = pd.DataFrame({'ds': pd.to_datetime(group['Timestamp']), 'y': group['Value']})

            if len(df_prophet) < 5:  # Minimum data points check
                st.warning(f"Not enough anomaly data points for {school} - {metric} to perform trend analysis.")
                continue

            try:
                model = Prophet(uncertainty_samples=False)  # Suppress uncertainty
                model.fit(df_prophet)

                future = model.make_future_dataframe(periods=7, freq='D')  # Adjust periods
                forecast = model.predict(future)

                if np.isnan(forecast['yhat']).any():  # NaN check
                    st.warning(f"Forecast for {school} - {metric} contains NaN values; trend analysis may be unreliable.")
                else:
                    st.line_chart(forecast[['ds', 'yhat']].set_index('ds'))

                    st.write("Interpretation:")
                    if forecast['yhat'].iloc[-1] > forecast['yhat'].iloc[0] * 1.2:
                        st.write("Increasing trend of anomalies for this metric in this school.")
                    elif forecast['yhat'].iloc[-1] < forecast['yhat'].iloc[0] * 0.8:
                        st.write("Decreasing trend of anomalies for this metric in this school.")
                    else:
                        st.write("Stable trend of anomalies for this metric in this school.")

            except Exception as e:
                st.error(f"Error during trend analysis for {school} - {metric}: {e}")

    else:
        st.write("No anomalies reported for trend analysis.")
