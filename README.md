---

# NILM - Non-Intrusive Load Monitoring for Smart Grid Energy Disaggregation

This project develops a **machine learning system for Non-Intrusive Load Monitoring (NILM)**. It predicts the operational state (ON/OFF) of individual appliances from aggregate electricity consumption data.

It includes:

* Time-series data analysis and feature engineering
* Multi-output classification for appliance ON/OFF detection
* Model evaluation (accuracy, precision, recall, F1-score)
* Energy consumption pattern visualization
* Interactive Streamlit application for real-time appliance predictions

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [Dataset Preparation](#dataset-preparation)
3. [Installation](#installation)
4. [Training the Model](#training-the-model)
5. [Evaluating the Model](#evaluating-the-model)
6. [Visualizing the Results](#visualizing-the-results)
7. [Running the Streamlit App](#running-the-streamlit-app)
8. [Using Synthetic Data](#using-synthetic-data)
9. [Notes](#notes)

---

## Project Structure

```
NILM-Project/
│
├─ app/                
│   └─ app.py          # Streamlit interface
├─ data/               
│   └─ *.csv           # iAWE dataset or synthetic data
├─ src/                
│   ├─ feature_engineering.py
│   ├─ train.py
│   ├─ evaluate.py
│   └─ visualize.py
├─ nilm_model.pkl      # Trained model after training
├─ requirements.txt
└─ README.md
```

---

## Dataset Preparation

### Using iAWE Dataset

1. Download `electricity.tar.gz` from [iAWE Dataset](http://i-awe.org/).
2. Extract the CSV files.
3. Place them in the `data/` folder:

```
data/
├─ aggregate.csv
├─ fridge.csv
├─ ac.csv
└─ washing_machine.csv
```

### Using Synthetic Data

The project can generate synthetic data for testing using `load_synthetic_data()`.

---

## Installation

Install required Python libraries:

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install pandas numpy scikit-learn matplotlib streamlit joblib
```

---

## Training the Model

Train the NILM model:

```bash
python -m src.train
```

This will create `nilm_model.pkl` in the project folder.

---

## Evaluating the Model

Evaluate model performance metrics:

```bash
python -m src.evaluate
```

Metrics include **accuracy, precision, recall, and F1-score** for each appliance.

---

## Visualizing the Results

Generate plots to visualize disaggregated appliance usage:

```bash
python -m src.visualize
```

---

## Running the Streamlit App

Launch the interactive app:

```bash
python -m streamlit run app/app.py
```

* Select appliances to display ON/OFF states
* Plot shows aggregate power (black) and appliance states (colored lines)

---

## Using Synthetic Data

* By default, the app uses synthetic data.
* To switch to iAWE dataset, modify `app.py`:

```python
from src.feature_engineering import load_data, create_features
data = load_data()  # loads iAWE CSVs
```

* You can also increase synthetic data samples:

```python
data = load_synthetic_data(n_samples=5000)
```

