import streamlit as st
from src.feature_engineering import load_data, create_features
import joblib
import matplotlib.pyplot as plt
import pandas as pd

st.title("NILM Energy Disaggregation - ON/OFF")

data = load_data()
df_feat = create_features(data)
model = joblib.load('nilm_model.pkl')

X = df_feat[['power','hour','minute','dayofweek','rolling_mean_5','rolling_std_5']]
y_pred = model.predict(X)
df_pred = pd.DataFrame(y_pred, columns=['fridge','ac','washing_machine'], index=df_feat.index)

st.subheader("Appliance ON/OFF Predictions")
st.dataframe(df_pred.head(20))

st.subheader("Energy Disaggregation Plot")
fig, ax = plt.subplots(figsize=(12,6))
ax.plot(df_feat.index, df_feat['power'], label='Aggregate Power', color='black', alpha=0.5)
ax.step(df_pred.index, df_pred['fridge'], label='Fridge (ON/OFF)', color='blue', where='post')
ax.step(df_pred.index, df_pred['ac'], label='AC (ON/OFF)', color='red', where='post')
ax.step(df_pred.index, df_pred['washing_machine'], label='Washing Machine (ON/OFF)', color='green', where='post')
ax.set_xlabel("Time")
ax.set_ylabel("State (0=OFF,1=ON)")
ax.legend()
st.pyplot(fig)
