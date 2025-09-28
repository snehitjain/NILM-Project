import pandas as pd
import numpy as np

def load_synthetic_data(n_samples=1000):
    """Generate synthetic NILM dataset with realistic appliance contributions."""
    timestamps = pd.date_range(start='2025-01-01', periods=n_samples, freq='T')

    # Appliances with higher ON probability
    fridge = np.random.choice([0, 100, 150], size=n_samples, p=[0.5, 0.3, 0.2])
    ac = np.random.choice([0, 200, 250], size=n_samples, p=[0.6, 0.25, 0.15])
    washing_machine = np.random.choice([0, 300, 350], size=n_samples, p=[0.7, 0.2, 0.1])

    # Aggregate includes all appliances + small noise
    aggregate = fridge + ac + washing_machine + np.random.randint(0, 20, size=n_samples)

    data = {
        'aggregate': pd.DataFrame({'power': aggregate}, index=timestamps),
        'fridge': pd.DataFrame({'power': fridge}, index=timestamps),
        'ac': pd.DataFrame({'power': ac}, index=timestamps),
        'washing_machine': pd.DataFrame({'power': washing_machine}, index=timestamps)
    }
    return data

def create_features(data):
    """Create features from aggregate power."""
    df = data['aggregate'].copy()
    df['hour'] = df.index.hour
    df['minute'] = df.index.minute
    df['dayofweek'] = df.index.dayofweek
    df['rolling_mean_5'] = df['power'].rolling(window=5).mean().fillna(method='bfill')
    df['rolling_std_5'] = df['power'].rolling(window=5).std().fillna(method='bfill')
    return df

def create_labels(data):
    """Create ON/OFF labels for appliances."""
    y = pd.DataFrame({
        'fridge': (data['fridge']['power'] > 50).astype(int),
        'ac': (data['ac']['power'] > 100).astype(int),
        'washing_machine': (data['washing_machine']['power'] > 100).astype(int)
    }, index=data['aggregate'].index)
    return y
