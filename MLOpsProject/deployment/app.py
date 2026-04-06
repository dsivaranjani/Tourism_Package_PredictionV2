import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="RanjaniD/Tourism_Package_PredictionV2", filename="best_Tourism_Package_Prediction_model_v1.joblib")
model = joblib.load(model_path)

# UI
st.set_page_config(page_title="Tourism Package Prediction", layout="centered")

st.title("🌍 Wellness Tourism Package Prediction")
st.write("""
Predict whether a customer is likely to purchase the **Wellness Tourism Package**.
Fill in the customer details below.
""")

# -------------------------
# USER INPUTS
# -------------------------

age = st.slider("Age", 18, 80, 30)

typeofcontact = st.selectbox(
    "Type of Contact",
    ["Company Invited", "Self Inquiry"]
)

citytier = st.selectbox("City Tier", [1, 2, 3])

occupation = st.selectbox(
    "Occupation",
    ["Salaried", "Freelancer", "Small Business", "Large Business"]
)

gender = st.selectbox("Gender", ["Male", "Female"])

num_persons = st.number_input("Number of Persons Visiting", 1, 10, 2)

preferred_star = st.selectbox("Preferred Hotel Rating", [3, 4, 5])

marital_status = st.selectbox(
    "Marital Status",
    ["Single", "Married", "Divorced"]
)

num_trips = st.number_input("Number of Trips per Year", 0, 20, 2)

passport = st.selectbox("Has Passport?", [0, 1])
own_car = st.selectbox("Owns Car?", [0, 1])

children = st.number_input("Children Traveling", 0, 5, 0)

designation = st.selectbox(
    "Designation",
    ["Executive", "Manager", "Senior Manager", "AVP", "VP"]
)

monthly_income = st.number_input("Monthly Income", 1000, 1000000, 30000)

pitch_score = st.slider("Pitch Satisfaction Score", 1, 5, 3)

product_pitched = st.selectbox(
    "Product Pitched",
    ["Basic", "Standard", "Deluxe", "Super Deluxe"]
)

followups = st.number_input("Number of Follow-ups", 0, 10, 2)

pitch_duration = st.number_input("Pitch Duration (minutes)", 5, 60, 20)

# -------------------------
# CREATE INPUT DATAFRAME
# -------------------------

input_data = pd.DataFrame([{
    "Age": age,
    "TypeofContact": typeofcontact,
    "CityTier": citytier,
    "Occupation": occupation,
    "Gender": gender,
    "NumberOfPersonVisiting": num_persons,
    "PreferredPropertyStar": preferred_star,
    "MaritalStatus": marital_status,
    "NumberOfTrips": num_trips,
    "Passport": passport,
    "OwnCar": own_car,
    "NumberOfChildrenVisiting": children,
    "Designation": designation,
    "MonthlyIncome": monthly_income,
    "PitchSatisfactionScore": pitch_score,
    "ProductPitched": product_pitched,
    "NumberOfFollowups": followups,
    "DurationOfPitch": pitch_duration
}])

# -------------------------
# PREDICTION
# -------------------------

if st.button("Predict Purchase"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.success(f"✅ Likely to Purchase (Probability: {probability:.2f})")
    else:
        st.error(f"❌ Not Likely to Purchase (Probability: {probability:.2f})")
