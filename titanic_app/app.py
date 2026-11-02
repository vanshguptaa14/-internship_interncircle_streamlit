import joblib
import pandas as pd
import streamlit as st
from pathlib import Path

# Same column order as used when the model was trained
FEATURES = ["pclass", "sex", "age", "fare"]

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢")


def find_model_file():
    """Look next to app.py first, then in the current working folder."""
    candidates = []
    if "__file__" in globals():
        candidates.append(Path(__file__).resolve().parent / "model.joblib")
    candidates.append(Path("model.joblib"))
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("model.joblib not found")


@st.cache_resource
def load_model():
    """Load the saved model once, so it is not reloaded on every click."""
    return joblib.load(find_model_file())


st.title("Would this passenger have survived?")
st.write(
    "An educational machine learning example: a decision tree trained on "
    "the Titanic passenger list. Choose the passenger details and press Predict."
)

col1, col2 = st.columns(2)
with col1:
    pclass = st.selectbox("Ticket class", [1, 2, 3], index=2,
                          help="1 = upper, 2 = middle, 3 = lower class")
    sex = st.radio("Sex", ["female", "male"], horizontal=True)
with col2:
    age = st.slider("Age (years)", 1, 80, 30)
    fare = st.slider("Fare paid (pounds)", 0, 500, 30)

if st.button("Predict", type="primary"):
    try:
        model = load_model()
    except FileNotFoundError:
        st.error("The model file (model.joblib) is missing. "
                 "Train and save it first, then put it next to app.py.")
        st.stop()
    except Exception:
        st.error("The model could not be loaded. It may have been saved with "
                 "a different scikit-learn version than the one installed here.")
        st.stop()

    try:
        row = pd.DataFrame(
            [[pclass, 1 if sex == "female" else 0, age, fare]],
            columns=FEATURES,
        )
        chance = float(model.predict_proba(row)[0][1])
    except Exception:
        st.error("Something went wrong while predicting. "
                 "Please check the inputs and try again.")
        st.stop()

    st.metric("Estimated chance of survival", f"{chance:.0%}")
    st.progress(chance)
    if chance >= 0.5:
        st.success("The model predicts this passenger likely survived.")
    else:
        st.warning("The model predicts this passenger likely did not survive.")

st.caption(
    "About this app: a small decision tree (max depth 4) trained on Titanic "
    "passengers with a known age, using class, sex, age and fare. It describes "
    "a historical dataset and shows a pattern, not a cause. A learning "
    "exercise, not advice. No personal data is collected."
)
