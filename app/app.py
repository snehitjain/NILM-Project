import streamlit as st
from src.feature_engineering import load_synthetic_data, create_features
import joblib
import matplotlib.pyplot as plt
import pandas as pd

st.title("NILM Energy Disaggregation - ON/OFF")

# Load data and model
data = load_synthetic_data()
df_feat = create_features(data)
model = joblib.load('nilm_model.pkl')

# Predict
X = df_feat[['power','hour','minute','dayofweek','rolling_mean_5','rolling_std_5']]
y_pred = model.predict(X)
df_pred = pd.DataFrame(y_pred, columns=['fridge','ac','washing_machine'], index=df_feat.index)

# Appliance selection
appliances = st.multiselect("Select appliances to display:", ['fridge','ac','washing_machine'], default=['fridge','ac','washing_machine'])

# Show all if none selected
if not appliances:
    appliances = ['fridge','ac','washing_machine']

st.subheader("Appliance ON/OFF Predictions")
st.dataframe(df_pred[appliances].head(20))

st.subheader("Energy Disaggregation Plot")
fig, ax = plt.subplots(figsize=(12,6))
ax.plot(df_feat.index, df_feat['power'], label='Aggregate Power', color='black', alpha=0.5)
colors = {'fridge':'blue', 'ac':'red', 'washing_machine':'green'}

for app in appliances:
    ax.step(df_pred.index, df_pred[app], label=f"{app.capitalize()} (ON/OFF)", color=colors[app], where='post')

ax.set_xlabel("Time")
ax.set_ylabel("State (0=OFF,1=ON)")
ax.legend()
st.pyplot(fig)
