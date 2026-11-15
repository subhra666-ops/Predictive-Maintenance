import streamlit as st, joblib, pandas as pd

model = joblib.load("model.pkl")
cols = joblib.load("columns.pkl")
threshold = joblib.load("threshold.pkl")

st.title("Machine Failure Predictor")
t = st.selectbox("Product type", ["L", "M", "H"])
air = st.number_input("Air temperature [K]", 290.0, 310.0, 300.0)
proc = st.number_input("Process temperature [K]", 300.0, 320.0, 310.0)
rpm = st.number_input("Rotational speed [rpm]", 1000, 3000, 1500)
torque = st.number_input("Torque [Nm]", 0.0, 100.0, 40.0)
wear = st.number_input("Tool wear [min]", 0, 300, 100)

if st.button("Predict"):
    row = {"air_temp": air, "process_temp": proc, "rpm": rpm,
           "torque": torque, "tool_wear": wear,
           "temp_diff": proc - air,
           "power": torque * rpm * 2 * 3.1416 / 60,
           "wear_torque": wear * torque,
           "type_L": int(t == "L"), "type_M": int(t == "M")}
    X_new = pd.DataFrame([row]).reindex(columns=cols, fill_value=0)
    prob = model.predict_proba(X_new)[0, 1]
    st.metric("Failure probability", f"{prob:.1%}")
    if prob >= threshold:
        st.error("High failure risk: schedule maintenance")
    else:
        st.success("Machine looks healthy")
