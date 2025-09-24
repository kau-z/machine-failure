import streamlit as st
import pandas as pd
import joblib, json
from pathlib import Path
import shap

# ── Locate model & threshold ─────────────────────────────
root = Path(__file__).resolve().parent.parent
model_path    = root / "notebooks" / "best_model_pipeline.joblib"
threshold_path = root / "notebooks" / "decision_threshold.json"

# Check required files
for p in [model_path, threshold_path]:
    if not p.exists():
        st.error(f"Required file missing: {p}")
        st.stop()

# ── Load binary failure model & threshold ────────────────
pipe = joblib.load(model_path)
with open(threshold_path) as f:
    decision_threshold = json.load(f)["threshold"]

# Grab trained estimator & preprocessor for SHAP
rf   = pipe.named_steps["model"]
prep = pipe.named_steps["prep"]
explainer = shap.TreeExplainer(rf)

# ── Streamlit UI ─────────────────────────────────────────
st.title("Machine Failure Predictor with Reason (Rule + SHAP)")

st.markdown(
    """
    Enter machine operating parameters to estimate the probability of failure.  
    If a failure is predicted, the app shows:
    * **Rule-based reasons** – simple checks on sensor thresholds  
    * **SHAP feature importances** – top features influencing the prediction
    """
)

# Input fields
t    = st.selectbox("Type", ["L", "M", "H"])
air  = st.number_input("Air temperature [K]", 250.0, 400.0, 300.0)
proc = st.number_input("Process temperature [K]", 250.0, 450.0, 310.0)
rpm  = st.number_input("Rotational speed [rpm]", 0, 3000, 1500)
tor  = st.number_input("Torque [Nm]", 0.0, 100.0, 40.0)
wear = st.number_input("Tool wear [min]", 0, 300, 120)

# Build a single-row DataFrame matching training features
row = pd.DataFrame([{
    "Type": t,
    "Air temperature [K]": air,
    "Process temperature [K]": proc,
    "Rotational speed [rpm]": rpm,
    "Torque [Nm]": tor,
    "Tool wear [min]": wear,
    "Temp diff [K]": proc - air,  # engineered feature used in training
}])

# Simple rule-based reasoning
def rule_based_reason(r):
    reasons = []
    temp_diff = r["Process temperature [K]"].iloc[0] - r["Air temperature [K]"].iloc[0]
    if temp_diff > 80:
        reasons.append("High process–air temperature difference (possible heat-dissipation issue)")
    if r["Tool wear [min]"].iloc[0] > 200:
        reasons.append("Excessive tool wear")
    if r["Torque [Nm]"].iloc[0] > 60:
        reasons.append("High torque / overstrain")
    if r["Rotational speed [rpm]"].iloc[0] > 2000:
        reasons.append("High rotational speed")
    return reasons or ["No specific rule triggered"]

# ── Prediction ───────────────────────────────────────────
if st.button("Predict"):
    # 1. Overall failure probability
    prob = pipe.predict_proba(row)[0, 1]
    st.metric("Failure probability", f"{prob:.3f}")

    if prob >= decision_threshold:
        st.write("Decision:", "⚠️ **Fail (1)**")

        # --- Rule-based reasons ---
        st.subheader("Rule-based reason(s):")
        st.write(", ".join(rule_based_reason(row)))

        # --- SHAP top feature contributions ---
        st.subheader("Top contributing features (SHAP):")
        X_trans = prep.transform(row)
        shap_values = explainer.shap_values(X_trans)

        # Handle different SHAP output shapes
        if isinstance(shap_values, list):
            # Standard binary: [class0, class1]
            shap_row = shap_values[1][0]
        elif shap_values.ndim == 3 and shap_values.shape[-1] == 2:
            # Single array with shape (1, n_features, 2)
            shap_row = shap_values[0, :, 1]
        else:
            # Already shape (1, n_features)
            shap_row = shap_values[0]

        # Build importance Series
        feat_imp = (
            pd.Series(shap_row, index=prep.get_feature_names_out())
              .abs()
              .sort_values(ascending=False)
        )
        st.write(feat_imp.head(5).to_frame("SHAP importance"))

    else:
        st.write("Decision:", "✅ No Fail (0)")
