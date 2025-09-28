# src/evaluate.py
from src.feature_engineering import load_synthetic_data, create_features, create_labels
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import matplotlib.pyplot as plt

# Load synthetic data
data = load_synthetic_data(n_samples=2000)
X = create_features(data)
y = create_labels(data)

# Load trained model
model = joblib.load('model.pkl')

# Predict
y_pred = model.predict(X)

# Metrics
for i, col in enumerate(y.columns):
    acc = accuracy_score(y[col], y_pred[:, i])
    prec = precision_score(y[col], y_pred[:, i], zero_division=0)
    rec = recall_score(y[col], y_pred[:, i], zero_division=0)
    f1 = f1_score(y[col], y_pred[:, i], zero_division=0)
    print(f"{col.capitalize()} Accuracy: {acc:.3f}")
    print(f"{col.capitalize()} Precision: {prec:.3f}")
    print(f"{col.capitalize()} Recall: {rec:.3f}")
    print(f"{col.capitalize()} F1: {f1:.3f}\n")

# Combined plot for all appliances (first 200 samples)
plt.figure(figsize=(15, 6))
colors = {'fridge':'blue', 'ac':'red', 'washing_machine':'green'}
markers = {'actual':'o', 'predicted':'x'}

for col in y.columns:
    plt.plot(y[col].values[:200], label=f'{col} actual', marker=markers['actual'], linestyle='-', color=colors[col])
    plt.plot(y_pred[:, y.columns.get_loc(col)][:200], label=f'{col} predicted', marker=markers['predicted'], linestyle='--', color=colors[col])

plt.title("Appliance ON/OFF States (first 200 samples)")
plt.xlabel("Time step")
plt.ylabel("State (0=OFF, 1=ON)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
