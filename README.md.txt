# Predictive Maintenance: Machine Failure Prediction

## Live Demo
[Open the app](https://machine-failure-predictorr.streamlit.app/)

Machine learning project that predicts machine failure from sensor data
using the AI4I 2020 dataset (UCI, 10,000 rows, about 3.4% failures).

## What's inside
- `MLproject.ipynb`: full pipeline (EDA, feature engineering, modelling, tuning, SHAP)
- `app.py`: Streamlit demo app
- `model.pkl`, `columns.pkl`, `threshold.pkl`: trained model, feature list, decision threshold
- `requirements.txt`: Python dependencies

## Method
- Features: air/process temperature, rpm, torque, tool wear, product type
- Engineered: temperature difference, power, wear x torque
- Imbalance handled with class weights; XGBoost tuned with 5-fold cross-validation
- Threshold chosen on training data to balance precision and recall
- Evaluated with PR-AUC, precision, recall, F1; explained with SHAP

## Run it
```
pip install -r requirements.txt
streamlit run app.py
```

## Dataset
AI4I 2020 Predictive Maintenance Dataset, UCI Machine Learning Repository
(https://archive.ics.uci.edu/dataset/601), CC BY 4.0.
