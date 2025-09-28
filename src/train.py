from src.feature_engineering import load_data, create_features
from src.model import build_model
from sklearn.model_selection import train_test_split
import joblib
import pandas as pd

data = load_data()
df_feat = create_features(data)

X = df_feat[['power', 'hour', 'minute', 'dayofweek', 'rolling_mean_5', 'rolling_std_5']]
y = pd.DataFrame({
    'fridge': (data['fridge']['power'] > 0).astype(int),
    'ac': (data['ac']['power'] > 0).astype(int),
    'washing_machine': (data['washing_machine']['power'] > 0).astype(int)
})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = build_model()
model.fit(X_train, y_train)

joblib.dump(model, 'nilm_model.pkl')
print("Model trained and saved as nilm_model.pkl")
