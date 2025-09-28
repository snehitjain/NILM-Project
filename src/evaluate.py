# src/evaluate.py
from src.feature_engineering import load_synthetic_data, create_features, create_labels
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import os

# ---------------------------
# Load synthetic data
# ---------------------------
data = load_synthetic_data(n_samples=2000)
X = create_features(data)
y = create_labels(data)

# ---------------------------
# Load trained model
# ---------------------------
model_file = "model.pkl"  # Ensure this is the trained model file
if not os.path.exists(model_file):
    raise FileNotFoundError(f"{model_file} not found. Train the model first.")

model = joblib.load(model_file)

# ---------------------------
# Predict
# ---------------------------
y_pred = model.predict(X)

# ---------------------------
# Compute metrics
# ---------------------------
metrics_list = []

for i, col in enumerate(y.columns):
    acc = accuracy_score(y[col], y_pred[:, i])
    prec = precision_score(y[col], y_pred[:, i], zero_division=0)
    rec = recall_score(y[col], y_pred[:, i], zero_division=0)
    f1 = f1_score(y[col], y_pred[:, i], zero_division=0)
    
    metrics_list.append({
        'Appliance': col.capitalize(),
        'Accuracy': round(acc, 3),
        'Precision': round(prec, 3),
        'Recall': round(rec, 3),
        'F1-Score': round(f1, 3)
    })

metrics_df = pd.DataFrame(metrics_list)
metrics_df.to_csv("evaluation_metrics.csv", index=False)
print("Evaluation Metrics:\n", metrics_df)

# ---------------------------
# Plot predicted vs actual (first 200 samples)
# ---------------------------
plt.figure(figsize=(15, 6))
colors = {'fridge':'blue', 'ac':'red', 'washing_machine':'green'}
markers = {'actual':'o', 'predicted':'x'}

for col in y.columns:
    plt.plot(y[col].values[:200], label=f'{col.capitalize()} Actual', marker=markers['actual'],
             linestyle='-', color=colors[col])
    plt.plot(y_pred[:, y.columns.get_loc(col)][:200], label=f'{col.capitalize()} Predicted',
             marker=markers['predicted'], linestyle='--', color=colors[col])

plt.title("Appliance ON/OFF States (First 200 Samples)")
plt.xlabel("Time Step")
plt.ylabel("State (0=OFF, 1=ON)")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save plot for report
plot_file = "appliance_onoff_plot.png"
plt.savefig(plot_file)
print(f"Plot saved as {plot_file}")

# Show plot
plt.show()
