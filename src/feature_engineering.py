import pandas as pd

def load_data():
    appliances = ['aggregate', 'fridge', 'ac', 'washing_machine']
    data = {}
    for app in appliances:
        data[app] = pd.read_csv(f'data/{app}.csv', parse_dates=['timestamp'])
        data[app].set_index('timestamp', inplace=True)
    return data

def create_features(data):
    df = data['aggregate'].copy()
    df['hour'] = df.index.hour
    df['minute'] = df.index.minute
    df['dayofweek'] = df.index.dayofweek
    df['rolling_mean_5'] = df['power'].rolling(window=5).mean().fillna(method='bfill')
    df['rolling_std_5'] = df['power'].rolling(window=5).std().fillna(method='bfill')
    return df
