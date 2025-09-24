import streamlit as st
import pandas as pd
import joblib, json
from pathlib import Path

# ── Locate models & threshold ─────────────────────────────
here = Path(__file__).resolve().parent
root = here.parent

model_path      = root / "notebooks" / "best_model_pipeline.joblib"
reason_model_path = root / "notebooks" / "failure_reason_model.joblib"
threshold_path  = root / "notebooks" / "decision_threshold.json"

# Check files
for p in [model_path, reason_model_path, threshold_path]:
    if not p.exists():
        st.error(f"Required file missing: {p}")
        st.stop()

# ── Load models ───────────────────────────────────────────
pipe = joblib.load(model_path)                # binary failure model
reason_pipe = joblib.load(reason_model_path)  # multi-label reason model
with open(threshold_path) as f:
    th = json.load(f)["threshold"]

# ── Streamlit UI ──────────────────────────────────────────
st.title("Machine Failure Predictor with Reason")

t   = st.selectbox("Type", ["L", "M", "H"])
air = st.number_input("Air temperature [K]", 250.0, 400.0, 300.0)
proc= st.number_input("Process temperature [K]", 250.0, 450.0, 310.0)
rpm = st.number_input("Rotational speed [rpm]", 0, 3000, 1500)
tor = st.number_input("Torque [Nm]", 0.0, 100.0, 40.0)
wear= st.number_input("Tool wear [min]", 0, 300, 120)

# Build input row
row = pd.DataFrame([{
    "Type": t,
    "Air temperature [K]": air,
    "Process temperature [K]": proc,
    "Rotational speed [rpm]": rpm,
    "Torque [Nm]": tor,
    "Tool wear [min]": wear,
    "Temp diff [K]": proc - air,   # engineered feature
}])

# ── Prediction button ─────────────────────────────────────
if st.button("Predict"):
    p = pipe.predict_proba(row)[0, 1]
    st.metric("Failure probability", f"{p:.3f}")

    if p >= th:
        st.write("Decision:", "⚠️ Fail (1)")
        # ----- Predict reason(s) -----
        reason_preds = reason_pipe.predict(row)[0]
        reason_labels = ["Tool Wear Failure", "Heat Dissipation Failure",
                         "Power Failure", "Overstrain Failure", "Random Failure"]
        active_reasons = [lbl for lbl, flag in zip(reason_labels, reason_preds) if flag == 1]
        if active_reasons:
            st.subheader("Likely reason(s) for failure:")
            st.write(", ".join(active_reasons))
        else:
            st.subheader("Likely reason:")
            st.write("Random/unspecified")
    else:
        st.write("Decision:", "✅ No Fail (0)")
