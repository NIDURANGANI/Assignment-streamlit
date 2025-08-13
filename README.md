
# 🚦 Road Accident Analysis & Prediction App

This is a **Streamlit-based Machine Learning application** designed to analyze road accident data in Sri Lanka and predict total accidents based on historical data. The app provides interactive data exploration, visualizations, and a predictive model for total accidents.

---

## 📌 Features

### Home
- App introduction and dataset overview.
- Banner image included.
- Download the full dataset as CSV.

### Data Exploration
- View dataset structure, sample data, and summary statistics.
- Filter data by Location.
- Dynamic column selection.

### Visualisations
- Total Accidents by Location (Bar chart).
- Deaths Distribution (Histogram).
- Grievous vs Non-Grievous Injuries (Comparative bar chart).
- Accident Map using realistic Sri Lanka coordinates.

### Model Prediction
- User input form for accident details.
- Display input values in a table.
- Predict total accidents with a trained machine learning model.
- Download predicted value as CSV.

### Model Performance
- Compare model R² scores for different algorithms.
- Visual performance charts.

---

## 🗂 Project Structure

```

road-accident-app/
├── app.py                  # Streamlit application
├── requirements.txt        # Dependencies
├── best\_model.pkl          # Trained ML model
├── scaler.pkl              # StandardScaler
├── data/
│   └── road\_accident.csv   # Dataset
├── notebooks/
│   └── model\_training.ipynb # Model training notebook
└── README.md               # Project description

````

---

## 🛠 Installation & Run Locally

1. Clone the repository:

```bash
git clone <your-repo-url>
cd road-accident-app
````

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
streamlit run app.py
```

---

## 📊 Model & Methodology

* **Model Used:** Random Forest Regressor (best R² = 0.85)
* **Preprocessing:** StandardScaler for numerical features.
* **Features:** Deaths (Male/Female), Grievous Injuries, Non-Grievous Injuries.
* **Target:** Total Accidents.

---

## 🌐 Deployment

* Deployed on **Streamlit Cloud**.
* Repository is connected and app runs live.
* Public URL: \[Insert Streamlit Cloud link here]

---

## ⚡ Challenges & Learnings

* Handling interactive maps and visualizations.
* Ensuring model prediction only runs after user input.
* Deploying Streamlit app with external dependencies.
* Learned end-to-end ML pipeline, data visualization, and cloud deployment.

---

## 📖 References

* [Streamlit Documentation](https://docs.streamlit.io/)
* Kaggle Road Accident Dataset

---

## 👩‍💻 Author
Kavindaya Nidurangani


