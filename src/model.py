from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

def build_model():
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    model = MultiOutputClassifier(rf)
    return model
