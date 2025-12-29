import streamlit as st
import pandas as pd
import joblib

# =========================
# Load trained model
# =========================
model = joblib.load("model.pkl")

st.title("✈️ Airline Customer Satisfaction Prediction")
st.markdown("### Enter Passenger Details")

# =========================
# Input fields
# =========================
gender = st.selectbox("Gender", ["Male", "Female"])
gender_num = 1 if gender == "Male" else 0
age = st.number_input("Age", 18, 100, 30)
travel_class = st.selectbox("Class", ["Eco", "Eco Plus", "Business"])
class_num = {"Eco": 0, "Eco Plus": 1, "Business": 2}[travel_class]

flight_distance = st.number_input("Flight Distance", 100, 5000, 800)
departure_delay = st.number_input("Departure Delay (minutes)", 0, 300, 0)
arrival_delay = st.number_input("Arrival Delay (minutes)", 0, 300, 0)

time_convenience = st.slider("Departure and Arrival Time Convenience", 0, 5, 3)
online_booking = st.slider("Ease of Online Booking", 0, 5, 3)
checkin = st.slider("Check-in Service", 0, 5, 3)
online_boarding = st.slider("Online Boarding", 0, 5, 3)
gate_location = st.slider("Gate Location", 0, 5, 3)
onboard_service = st.slider("On-board Service", 0, 5, 3)
seat_comfort = st.slider("Seat Comfort", 0, 5, 3)
legroom = st.slider("Leg Room Service", 0, 5, 3)
cleanliness = st.slider("Cleanliness", 0, 5, 3)
food_drink = st.slider("Food and Drink", 0, 5, 3)
inflight_service = st.slider("In-flight Service", 0, 5, 3)
inflight_wifi = st.slider("In-flight Wifi Service", 0, 5, 3)
entertainment = st.slider("In-flight Entertainment", 0, 5, 3)
baggage = st.slider("Baggage Handling", 0, 5, 3)

travel_type = st.selectbox("Type of Travel", ["Business", "Personal"])
customer_type = st.selectbox("Customer Type", ["Returning", "First-time"])

# =========================
# Build input DataFrame
# MUST match model.feature_names_in_
# =========================
input_data = pd.DataFrame([{
    "Gender": gender_num,
    "Age": age,
    "Class": class_num,
    "Flight Distance": flight_distance,
    "Departure Delay": departure_delay,
    "Arrival Delay": arrival_delay,
    "Departure and Arrival Time Convenience": time_convenience,
    "Ease of Online Booking": online_booking,
    "Check-in Service": checkin,
    "Online Boarding": online_boarding,
    "Gate Location": gate_location,
    "On-board Service": onboard_service,
    "Seat Comfort": seat_comfort,
    "Leg Room Service": legroom,
    "Cleanliness": cleanliness,
    "Food and Drink": food_drink,
    "In-flight Service": inflight_service,
    "In-flight Wifi Service": inflight_wifi,
    "In-flight Entertainment": entertainment,
    "Baggage Handling": baggage,
    "Type of Travel_Personal": 1 if travel_type == "Personal" else 0,
    "Customer Type_Returning": 1 if customer_type == "Returning" else 0
}])

# enforce correct column order
input_data = input_data[model.feature_names_in_]


# 🔑 VERY IMPORTANT: enforce column order
input_data = input_data[model.feature_names_in_]

# =========================
# Prediction
# =========================
if st.button("Predict Satisfaction"):
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)

    if prediction[0] == 1:
        st.success(f"😊 Passenger is **Satisfied**)")
    else:
        st.error(f"😐 Passenger is **Neutral or Dissatisfied*)")



