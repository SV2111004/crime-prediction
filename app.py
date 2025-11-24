import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Crime Prediction & Analysis", layout="wide")

# ===============================
# LOAD DATA
# ===============================
@st.cache_data
def load_data():
    return pd.read_csv("data/cleaned/final_merged_dataset.csv")

df = load_data()

# ===============================
# SIDEBAR NAVIGATION
# ===============================
menu = st.sidebar.radio(
    "Navigate",
    ["🏠 Home", "📁 Dataset Overview", "📊 EDA", "🔮 Prediction (Coming Soon)"]
)

# ===============================
# HOME
# ===============================
if menu == "🏠 Home":
    st.title("🔍 Crime Prediction & Analysis Dashboard")
    st.write("""
    Welcome to the India's Crime Analytics Platform.  
    Explore socio-economic factors and crime trends across states & crime categories.
    """)
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=180)
    st.write("Use the sidebar to navigate across different sections.")

# ===============================
# DATASET OVERVIEW
# ===============================
elif menu == "📁 Dataset Overview":
    st.title("📁 Dataset Overview")
    st.write("### Full Dataset")
    st.dataframe(df)
    st.write("### Summary Statistics")
    st.write(df.describe())
    st.write("### Missing Values")
    st.write(df.isnull().sum())

# ===============================
# INTERACTIVE EDA ONLY
# ===============================
elif menu == "📊 EDA":
    st.title("📊 Interactive Crime Data Analysis")

    import plotly.express as px

    crime_by_state = df.groupby("State")["Cases"].sum().sort_values(ascending=False).head(10)
    crime_by_type = df.groupby("Crime_Type")["Cases"].sum().sort_values(ascending=False)
    crime_by_year = df.groupby("Year")["Cases"].sum()

    st.subheader("1️⃣ Top 10 States with Highest Crime")
    fig = px.bar(
        crime_by_state,
        x=crime_by_state.values,
        y=crime_by_state.index,
        orientation="h",
        color=crime_by_state.values,
        title="Top 10 Crime-Affected States"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("2️⃣ Crime Type Distribution")
    fig = px.pie(
        names=crime_by_type.index,
        values=crime_by_type.values,
        title="Distribution of Crime Types"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("3️⃣ Crime Trend Over the Years")
    fig = px.line(
        x=crime_by_year.index,
        y=crime_by_year.values,
        markers=True,
        title="Year-Wise Crime Trend in India"
    )
    fig.update_xaxes(title="Year")
    fig.update_yaxes(title="Total Crime Cases")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("4️⃣ Correlation Heatmap of Socio-Economic Factors")
    st.write("Correlation between crime cases and socio-economic indicators")

    # Select only numerical columns
    num_df = df.select_dtypes(include=['float64', 'int64'])

    # Compute correlation
    corr = num_df.corr()

    # Interactive Heatmap
    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        title="Correlation Heatmap",
        aspect="auto"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("5️⃣ Crime Intensity Map of India")

    import requests
    import plotly.express as px

    # Raw geojson link
    geojson_url = "https://raw.githubusercontent.com/udit-001/india-maps-data/main/geojson/india.geojson"
    india_geo = requests.get(geojson_url).json()

    # Aggregate crime data
    crime_state_map = df.groupby("State")["Cases"].sum().reset_index()

    # Fix only the one mismatch
    crime_state_map["State"] = crime_state_map["State"].replace({
        "Andaman & Nicobar Islands": "Andaman & Nicobar"
    })

    fig = px.choropleth(
        crime_state_map,
        geojson=india_geo,
        featureidkey="properties.NAME_1",
        locations="State",
        color="Cases",
        color_continuous_scale="Reds",
        title="Crime Intensity Across Indian States"
    )

    fig.update_geos(fitbounds="locations", visible=False)
    st.plotly_chart(fig, use_container_width=True)



# ===============================
# PREDICTION PAGE (ON PAUSE)
# ===============================
elif menu == "🔮 Prediction (Coming Soon)":
    st.title("🔮 Prediction Section Coming Soon")
    st.write("""
    The crime prediction functionality (XGBoost + RF ensemble) will be added soon.

    🔹 Real-time crime forecasting  
    🔹 Risk-level categorization  
    🔹 Downloadable reports  

    Stay tuned! 🙂
    """)

# ===============================
st.write("---")
st.caption("© 2025 • Crime Prediction Project • Interactive Visualisation Version")
