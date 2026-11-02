# Titanic Survival Predictor (Streamlit)

A small interactive web app that estimates a passenger's chance of survival on the Titanic.
Built for InternCircle Project 05 (Data Science & AI) by Vansh Gupta.

- **Model:** decision tree (max depth 4) from scikit-learn, saved with joblib
- **Data:** Titanic passengers (seaborn dataset) with a known age; features: class, sex, age, fare
- **Test accuracy:** 79.7% (baseline 59.4%)
- **Run locally:** `pip install -r requirements.txt` then `streamlit run app.py`

This is an educational example. It describes a historical dataset and is not advice.
No personal data is collected.
