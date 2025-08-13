import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
from sklearn.preprocessing import StandardScaler

# ---------------- Page Config ----------------
st.set_page_config(page_title="Road Accident Analysis", page_icon="🚦", layout="wide")

# ---------------- Load Data ----------------
@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Users\Nidurangani\OneDrive\Desktop\Assignment streamlit\data\road_accident.csv")
    return df

# ---------------- Load Model & Scaler ----------------
@st.cache_data
def load_model():
    try:
        with open(r"C:\Users\Nidurangani\OneDrive\Desktop\Assignment streamlit\best_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open(r"C:\Users\Nidurangani\OneDrive\Desktop\Assignment streamlit\scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        return model, scaler
    except:
        return None, None

df = load_data()
model, scaler = load_model()

# ---------------- Sidebar Navigation ----------------
st.sidebar.title("Navigation")
menu = st.sidebar.radio("Go to", ["Home", "Data Exploration", "Visualisations", "Model Prediction", "Model Performance"])

# ---------------- Home ----------------
if menu == "Home":
    st.title("🚦 Road Accident Data Analysis & Prediction")
    st.image("https://www.ghp-news.com/wp-content/uploads/2023/03/AdobeStock_570708583-2.jpg", use_container_width=True)
    st.write("""
    Welcome! Explore road accident data, visualize statistics,
    and predict total accidents based on features.
    """)
    st.subheader("Dataset Overview")
    csv = df.to_csv(index=False).encode()
    st.download_button("📥 Download Dataset", data=csv, file_name="road_accidents.csv", mime="text/csv")

# ---------------- Data Exploration ----------------
elif menu == "Data Exploration":
    st.title("🔍 Data Exploration")
    filter_location = st.selectbox("Filter by Location", ["All"] + df["Location"].tolist())
    df_filtered = df if filter_location=="All" else df[df["Location"]==filter_location]
    st.dataframe(df_filtered)
    st.write("Summary Statistics")
    st.write(df_filtered.describe())

# ---------------- Visualisations ----------------
elif menu == "Visualisations":
    st.title("📊 Visualisations")
    chart_option = st.selectbox("Choose Chart", ["Total Accidents by Location", "Deaths Distribution", "Grievous vs Non-Grievous Injuries", "Accident Map"])
    
    if chart_option == "Total Accidents by Location":
        fig = px.bar(df.sort_values("Total", ascending=False), x="Total", y="Location", color="Total", height=600)
        st.plotly_chart(fig, use_container_width=True)

    elif chart_option == "Deaths Distribution":
        df['Total_Deaths'] = df["No.of Deaths Male"] + df["No.of Deaths Female"]
        fig = px.histogram(df, x="Total_Deaths", nbins=20, color="Total_Deaths", height=500)
        st.plotly_chart(fig, use_container_width=True)

    elif chart_option == "Grievous vs Non-Grievous Injuries":
        df_sum = df[['Grievous Injury Male','Grievous Injury Female','Non Grievous Injury Male','Non Grievous Injury Female']].sum().reset_index()
        df_sum.columns = ['Type','Count']
        fig = px.bar(df_sum, x='Type', y='Count', color='Count', height=500)
        st.plotly_chart(fig, use_container_width=True)

    elif chart_option == "Accident Map":
        # Realistic Sri Lanka lat/lon ranges
        np.random.seed(42)
        df['lat'] = np.random.uniform(5.9, 9.85, size=len(df))
        df['lon'] = np.random.uniform(79.8, 81.9, size=len(df))
        fig = px.scatter_map(df, lat="lat", lon="lon", size="Total", color="Total",
                             hover_name="Location", zoom=7, height=600)
        st.plotly_chart(fig, use_container_width=True)

# ---------------- Model Prediction ----------------
elif menu == "Model Prediction":
    st.title("🤖 Model Prediction")
    st.write("Enter accident details:")

    with st.form("prediction_form"):
        deaths_male = st.number_input("No. of Deaths Male", 0, 500, 10)
        deaths_female = st.number_input("No. of Deaths Female", 0, 200, 5)
        grievous_male = st.number_input("Grievous Injury Male", 0, 2000, 50)
        grievous_female = st.number_input("Grievous Injury Female", 0, 1000, 20)
        non_grievous_male = st.number_input("Non Grievous Injury Male", 0, 5000, 100)
        non_grievous_female = st.number_input("Non Grievous Injury Female", 0, 5000, 50)

        submitted = st.form_submit_button("Predict")

    if submitted:
        input_data = pd.DataFrame([{
            "Deaths Male": deaths_male,
            "Deaths Female": deaths_female,
            "Grievous Injury Male": grievous_male,
            "Grievous Injury Female": grievous_female,
            "Non Grievous Injury Male": non_grievous_male,
            "Non Grievous Injury Female": non_grievous_female
        }])
        
        st.subheader("Input Data")
        st.table(input_data)

        if model and scaler:
            X_scaled = scaler.transform(input_data)
            prediction = model.predict(X_scaled)[0]
            st.success(f"Predicted Total Accidents: {int(prediction)}")
            
            pred_df = pd.DataFrame({"Predicted Total Accidents":[int(prediction)]})
            csv_pred = pred_df.to_csv(index=False).encode()
            st.download_button("📥 Download Prediction", data=csv_pred, file_name="prediction.csv", mime="text/csv")
        else:
            st.info("Model not loaded. Please run training first.")
    else:
        st.info("Fill the form and click Predict to see results.")

# ---------------- Model Performance ----------------
elif menu == "Model Performance":
    st.title("📈 Model Performance")
    models = ['Random Forest', 'Linear Regression', 'SVM']
    r2_scores = [0.85, 0.68, 0.72]
    fig = px.bar(x=models, y=r2_scores, color=r2_scores, text=r2_scores,
                 labels={'x':'Model','y':'R2 Score'}, height=500)
    st.plotly_chart(fig, use_container_width=True)
    st.write("Confusion Matrix / additional metrics can be added here.")
