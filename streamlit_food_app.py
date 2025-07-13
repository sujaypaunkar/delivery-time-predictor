
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# --- Prepare Training Data ---
data = {
    'Distance_km': [2.0, 5.5, 3.0, 1.2, 4.0, 2.8, 6.0, 0.5],
    'Order_Size': [2, 4, 3, 1, 5, 2, 3, 1],
    'Traffic': ['Low', 'High', 'Medium', 'Low', 'High', 'Medium', 'High', 'Low'],
    'On_Time': ['Yes', 'No', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes']
}
df = pd.DataFrame(data)

# Encode
le_traffic = LabelEncoder()
df['Traffic_Encoded'] = le_traffic.fit_transform(df['Traffic'])
le_target = LabelEncoder()
df['On_Time_Encoded'] = le_target.fit_transform(df['On_Time'])

X = df[['Distance_km', 'Order_Size', 'Traffic_Encoded']]
y = df['On_Time_Encoded']

# Train
model = RandomForestClassifier()
model.fit(X, y)

# --- Streamlit UI ---
st.title("🚴‍♂️ Food Delivery On-Time Predictor")

distance = st.slider("📏 Distance to Customer (km)", 0.0, 10.0, 3.0)
order_size = st.slider("🍱 Order Size (Number of Items)", 1, 10, 2)
traffic = st.selectbox("🚦 Traffic Condition", ['Low', 'Medium', 'High'])

if st.button("Predict"):
    traffic_encoded = le_traffic.transform([traffic])[0]
    input_df = pd.DataFrame([[distance, order_size, traffic_encoded]], columns=['Distance_km', 'Order_Size', 'Traffic_Encoded'])
    pred = model.predict(input_df)[0]
    result = le_target.inverse_transform([pred])[0]
    st.success(f"🕒 Will the order be on time? → **{result}**")
