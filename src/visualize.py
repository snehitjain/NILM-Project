import matplotlib.pyplot as plt
from src.feature_engineering import load_synthetic_data, create_features
import joblib
import pandas as pd

data = load_synthetic_data()
df_feat = create_features(data)
model = joblib.load('nilm_model.pkl')

X = df_feat[['power', 'hour', 'minute', 'dayofweek', 'rolling_mean_5', 'rolling_std_5']]
y_pred = model.predict(X)
df_pred = pd.DataFrame(y_pred, columns=['fridge','ac','washing_machine'], index=df_feat.index)

plt.figure(figsize=(12,6))
plt.plot(df_feat.index, df_feat['power'], label='Aggregate Power', color='black', alpha=0.5)
plt.step(df_pred.index, df_pred['fridge'], label='Fridge (ON/OFF)', color='blue', where='post')
plt.step(df_pred.index, df_pred['ac'], label='AC (ON/OFF)', color='red', where='post')
plt.step(df_pred.index, df_pred['washing_machine'], label='Washing Machine (ON/OFF)', color='green', where='post')
plt.ylabel("State (0=OFF,1=ON)")
plt.xlabel("Time")
plt.title("Energy Disaggregation ON/OFF")
plt.legend()
plt.show()
plt.savefig("energy_disaggregation.png")

