from src.feature_engineering import load_synthetic_data, create_features, create_labels
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load synthetic data
data = load_synthetic_data(n_samples=2000)
X = create_features(data)[['power','hour','minute','dayofweek','rolling_mean_5','rolling_std_5']]
y = create_labels(data)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'nilm_model.pkl')
print("Model trained and saved!")
