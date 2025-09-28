from src.feature_engineering import load_data, create_features
import joblib
import pandas as pd
from sklearn.metrics import classification_report

model = joblib.load('nilm_model.pkl')
data = load_data()
df_feat = create_features(data)

X = df_feat[['power', 'hour', 'minute', 'dayofweek', 'rolling_mean_5', 'rolling_std_5']]
y = pd.DataFrame({
    'fridge': (data['fridge']['power'] > 0).astype(int),
    'ac': (data['ac']['power'] > 0).astype(int),
    'washing_machine': (data['washing_machine']['power'] > 0).astype(int)
})

y_pred = model.predict(X)
df_pred = pd.DataFrame(y_pred, columns=['fridge','ac','washing_machine'], index=df_feat.index)

print("Evaluation Report:")
print(classification_report(y, y_pred, target_names=['fridge','ac','washing_machine']))
