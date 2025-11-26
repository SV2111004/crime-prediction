import streamlit as st
import pandas as pd
import joblib
import json
import plotly.express as px

st.set_page_config(page_title="Crime Prediction Dashboard", layout="wide")

# LOAD DATA

@st.cache_data
def load_data():
    return pd.read_csv("data/cleaned/final_merged_dataset.csv")

df = load_data()

@st.cache_resource
def load_models():
    models = {
        "Decision Tree": joblib.load("model/decision_tree_final_depth13.pkl"),
        "Random Forest": joblib.load("model/random_forest_time_model.pkl")
    }
    preprocessor = joblib.load("model/decision_tree_preprocessor.pkl")
    return models, preprocessor

models, preprocessor = load_models()

# SIDE MENU

menu = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "📊 Visualizations", "📈 Predict Crime","📊 Model Comparison", "📌 Crime Hotspot Map"]
)

# HOME TAB
if menu == "🏠 Home":
    st.markdown("<h1>🔍 Crime Prediction & Analysis Dashboard</h1>", unsafe_allow_html=True)

    st.subheader("📌 Dataset Preview")
    st.dataframe(df)

    st.subheader("📌 Statistical Summary")
    st.dataframe(df.describe())

    st.markdown("### 🔥 Quick Insights")
    colA, colB, colC = st.columns(3)
    colA.metric("🔹 Total Records", df.shape[0])
    colB.metric("🗺️ Total States", df["State"].nunique())
    colC.metric("🚨 Crime Categories", df["Crime_Type"].nunique())

    st.write("""
    This dashboard:
    ✔ Visualizes crime patterns across India
    ✔ Shows crime hotspots using map
    ✔ Predicts crime count using machine learning models  
    """)

# VISUALIZATIONS TAB

elif menu == "📊 Visualizations":
    st.header("📊 Crime Trends & Statistical Visualizations")

    # ---- Top 10 Crime Prone States ----
    st.subheader("📌 Top 10 Crime-Prone States")
    top_states = df.groupby("State")["Cases"].sum().reset_index().sort_values(by="Cases", ascending=False).head(10)
    fig2 = px.bar(top_states, x="State", y="Cases", title="Top 10 States by Crime Cases")
    st.plotly_chart(fig2, use_container_width=True)

    # ---- Crime Types ----
    st.subheader("📌 Crime Types Comparison Across INDIA")
    top_crimes = df.groupby("Crime_Type")["Cases"].sum().reset_index().sort_values(by="Cases", ascending=False)
    fig3 = px.bar(top_crimes, x="Crime_Type", y="Cases", title="Crime Count per Crime Type", text_auto=True)
    st.plotly_chart(fig3, use_container_width=True)

    # ---- Pie Chart Crime Types ----
    fig4 = px.pie(top_crimes, names="Crime_Type", values="Cases", title="Contribution of Crime Types in Total Crime")
    st.plotly_chart(fig4, use_container_width=True)

    # ---- Boxplot ----
    st.subheader("⚖ Crime Case Distribution Across States")
    fig5 = px.box(df, x="State", y="Cases", title="Crime Case Distribution by State")
    st.plotly_chart(fig5, use_container_width=True)

    # ---- Trend over Years ----
    metric = st.selectbox("Select Metric", ["Cases", "unemployment_rate", "poverty_rate", "per_capita_income",
        "inflation_rate", "population_density", "gender_ratio",
        "literacy_rate", "youth_population_percent", "urbanization_rate",
        "human_development_index", "police_stations_per_district",
        "conviction_rate", "police_personnel_per_100k",
        "alcohol_consumption_per_capita"])
    yearly = df.groupby("Year")[metric].sum().reset_index()
    fig = px.line(yearly, x="Year", y=metric, markers=True, title=f"Trend of {metric} over Years")
    st.plotly_chart(fig, use_container_width=True)

# PREDICT CRIME 

elif menu == "📈 Predict Crime":
    st.header("📈 Crime Prediction")

    col1, col2 = st.columns(2)
    state = col1.selectbox("State", sorted(df["State"].unique()), key="pred_state")
    crime = col2.selectbox("Crime Type", sorted(df["Crime_Type"].unique()), key="pred_crime")
    year = col1.slider("Year", 2001, 2030, 2024, key="pred_year")

    num_features = [
        "unemployment_rate", "poverty_rate", "per_capita_income",
        "inflation_rate", "population_density", "gender_ratio",
        "literacy_rate", "youth_population_percent", "urbanization_rate",
        "human_development_index", "police_stations_per_district",
        "conviction_rate", "police_personnel_per_100k",
        "alcohol_consumption_per_capita"
    ]

    inputs = {"State": state, "Crime_Type": crime, "Year": year}
    for feat in num_features:
        inputs[feat] = st.number_input(feat, value=float(df[feat].mean()), key=f"num_{feat}")

    input_df = pd.DataFrame([inputs])

    # ---------- HYBRID PREDICTION USING 59-FEATURE MODELS ----------

    # 1) Load hybrid preprocessor + models + weights (saved from 08_hybrid.ipynb)
    hybrid_preprocessor = joblib.load("model/hybrid_preprocessor.pkl")
    xgb_model = joblib.load("model/xgb_hybrid.pkl")
    lgbm_model = joblib.load("model/lgbm_hybrid.pkl")
    rf_model  = joblib.load("model/rf_hybrid.pkl")
    best_combo = joblib.load("model/hybrid_weights.pkl")   # (wx, wl, wr)

    wx, wl, wr = best_combo  # XGB, LGBM, RF weights

    # 2) Transform input like in training (59 features)
    encoded = hybrid_preprocessor.transform(input_df)

    # 3) Get predictions from 3 models
    xgb_pred = xgb_model.predict(encoded)[0]
    lgbm_pred = lgbm_model.predict(encoded)[0]
    rf_pred  = rf_model.predict(encoded)[0]

    # 4) Hybrid weighted prediction (same as notebook)
    hybrid_pred = (wx * xgb_pred) + (wl * lgbm_pred) + (wr * rf_pred)

    # 5) Range: ±15% around hybrid
    lower = int(hybrid_pred * 0.85)
    upper = int(hybrid_pred * 1.15)

    st.success(f"🔮 Predicted Crime Cases Range (Hybrid): {lower:,} – {upper:,}")
    st.caption(f"Hybrid Weights → XGB: {wx:.2f}, LGBM: {wl:.2f}, RF: {wr:.2f}")

# Model Comparison 

elif menu == "📊 Model Comparison":
    st.header("📌 Machine Learning Model Performance Comparison")

    # Hardcoded scores
    scores = pd.DataFrame([
        {"Model": "Simple Regression", "R2": -0.0010, "RMSE": 7022.09, "MAE": 3300.09},
        {"Model": "Multiple Regression", "R2": 0.4490, "RMSE": 5210.02, "MAE": 2751.42},
        {"Model": "Decision Tree", "R2": 0.7342, "RMSE": 3618.53, "MAE": 1190.42},
        {"Model": "Random Forest", "R2": 0.8023, "RMSE": 3120.58, "MAE": 1059.69},
        {"Model": "LightGBM", "R2": 0.7650, "RMSE": 3401.94, "MAE": 1293.71},
        {"Model": "XGBoost", "R2": 0.8044, "RMSE": 3103.80, "MAE": 1010.34},
        {"Model": "Hybrid", "R2": 0.7976, "RMSE": 3157.08, "MAE": 1093.13}
    ])

    st.subheader("📋 Performance Table")
    st.dataframe(scores)

    # Accuracy Graph (R2)
    fig1 = px.bar(scores, x="Model", y="R2", color="Model",
                  title="R² Score — Higher is Better", text_auto=True)
    st.plotly_chart(fig1, use_container_width=True)

    # RMSE Graph
    fig2 = px.bar(scores, x="Model", y="RMSE", color="Model",
                  title="RMSE — Lower Error is Better", text_auto=True)
    st.plotly_chart(fig2, use_container_width=True)

    # MAE Graph
    fig3 = px.bar(scores, x="Model", y="MAE", color="Model",
                  title="MAE — Lower Error is Better", text_auto=True)
    st.plotly_chart(fig3, use_container_width=True)

    st.info("🔍 Conclusion: Random Forest performs best among the trained models.")

# CRIME HOTSPOT MAP

elif menu == "📌 Crime Hotspot Map":
    st.header("🌍 Crime Hotspots Across Indian States")

    with open("data/in.json", "r", encoding="utf-8") as f:
        india_geojson = json.load(f)

    year_filter = st.selectbox("Select Year", sorted(df["Year"].unique()), key="map_year")
    crime_filter = st.selectbox("Select Crime Type", sorted(df["Crime_Type"].unique()), key="map_crime")

    filtered = df[(df["Year"] == year_filter) & (df["Crime_Type"] == crime_filter)]
    crime_state_map = filtered.groupby("State").agg({
        "Cases": "sum",
        "literacy_rate": "mean",
        "human_development_index": "mean",
        "urbanization_rate": "mean"
    }).reset_index()

    if "Ladakh" not in crime_state_map["State"].values:
        jk_row = crime_state_map[crime_state_map["State"] == "Jammu & Kashmir"].iloc[0]
        crime_state_map = pd.concat([crime_state_map, pd.DataFrame([{
            "State": "Ladakh",
            "Cases": jk_row["Cases"],
            "literacy_rate": jk_row["literacy_rate"],
            "human_development_index": jk_row["human_development_index"],
            "urbanization_rate": jk_row["urbanization_rate"]
        }])], ignore_index=True)

    fig = px.choropleth(
        crime_state_map,
        geojson=india_geojson,
        featureidkey="properties.NAME_1",
        locations="State",
        color="Cases",
        hover_data=["Cases", "literacy_rate", "human_development_index", "urbanization_rate"],
        color_continuous_scale="Reds",
        title=f"Crime Hotspots — {crime_filter} ({year_filter})"
    )

    fig.update_geos(fitbounds="locations", visible=False)
    st.plotly_chart(fig, use_container_width=True)

st.markdown("<br><center>💡 Developed by Janvi | Akshat | Shubhra</center><br>", unsafe_allow_html=True)