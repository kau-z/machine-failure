# src/app.py
import streamlit as st
import pandas as pd
import joblib, json
from pathlib import Path
import plotly.graph_objects as go
import numpy as np

# ── Paths ────────────────────────────────────────────────
root = Path(__file__).resolve().parent.parent
model_path      = root / "notebooks" / "best_model_pipeline.joblib"
threshold_path  = root / "notebooks" / "decision_threshold.json"
ind_thresh_path = root / "notebooks" / "indicator_thresholds.json"   # ← data-driven rule cutoffs

# ── Load artifacts ───────────────────────────────────────
for p in [model_path, threshold_path]:
    if not p.exists():
        st.error(f"Required file missing: {p}")
        st.stop()

pipe = joblib.load(model_path)

with open(threshold_path) as f:
    decision_threshold = float(json.load(f)["threshold"])

# Indicator thresholds (learned from data). If not present, use defaults.
if ind_thresh_path.exists():
    with open(ind_thresh_path) as f:
        IND = json.load(f)
else:
    IND = {"HDF_dt_k": 80, "PWF_low": 3500, "PWF_high": 9000,
           "OSF_wear_torque": 12000, "TWF_wear": 200}

# ── Helpers ──────────────────────────────────────────────
def chip(label, ok=True):
    color = "#E6F4EA" if ok else "#FDE7E9"
    text  = "PASS" if ok else "ALERT"
    txtc  = "#137333" if ok else "#A50E0E"
    border = "#CEEAD6" if ok else "#F7C1C6"
    return f"""
    <span style="
        display:inline-block;padding:6px 12px;border-radius:9999px;
        background:{color};color:{txtc};font-weight:700;
        border:1px solid {border};margin:6px 8px 0 0;">
        {label}: {text}
    </span>
    """

def rule_indicators(row: pd.DataFrame):
    air  = row["Air temperature [K]"].iloc[0]
    proc = row["Process temperature [K]"].iloc[0]
    rpm  = row["Rotational speed [rpm]"].iloc[0]
    tor  = row["Torque [Nm]"].iloc[0]
    wear = row["Tool wear [min]"].iloc[0]
    dT   = proc - air
    power_index = tor * rpm
    wear_torque = wear * tor

    HDF_flag = dT > IND["HDF_dt_k"]
    PWF_flag = (power_index < IND["PWF_low"]) or (power_index > IND["PWF_high"])
    OSF_flag = wear_torque > IND["OSF_wear_torque"]
    TWF_flag = wear >= IND["TWF_wear"]

    # normalized intensities (0..1) for the bar chart
    hdf_int = np.clip((dT - IND["HDF_dt_k"]) / max(1.0, 0.4*IND["HDF_dt_k"]), 0, 1)
    band_w  = max(1.0, IND["PWF_high"] - IND["PWF_low"])
    if IND["PWF_low"] <= power_index <= IND["PWF_high"]:
        pwf_int = 0.0
    else:
        dist = IND["PWF_low"] - power_index if power_index < IND["PWF_low"] else power_index - IND["PWF_high"]
        pwf_int = np.clip(dist / (0.5 * band_w), 0, 1)
    osf_int = np.clip((wear_torque - IND["OSF_wear_torque"]) / max(1.0, 0.5*IND["OSF_wear_torque"]), 0, 1)
    twf_int = np.clip((wear - IND["TWF_wear"]) / max(1.0, 0.5*IND["TWF_wear"]), 0, 1)

    return {
        "flags": {"HDF": HDF_flag, "PWF": PWF_flag, "OSF": OSF_flag, "TWF": TWF_flag},
        "dT": dT, "power_index": power_index, "wear": wear,
        "intensities": {"HDF": hdf_int, "PWF": pwf_int, "OSF": osf_int, "TWF": twf_int}
    }

def reason_and_solution(flags: dict, fail: bool):
    reasons, actions = [], []
    f = flags["flags"]
    if fail:
        if f["HDF"]:
            reasons.append("Large process–air temperature difference (possible heat-dissipation issue).")
            actions.append("Reduce process temperature / improve cooling; verify airflow and sensors.")
        if f["PWF"]:
            reasons.append("Power outside expected band (torque×rpm proxy).")
            actions.append("Check load/drive; bring torque and rpm back into nominal band.")
        if f["OSF"]:
            reasons.append("Overstrain indication (high wear×torque).")
            actions.append("Lower load; inspect spindle/bearings; schedule maintenance.")
        if f["TWF"]:
            reasons.append("Excessive tool wear.")
            actions.append("Replace/rotate tool; review cutting parameters and lubrication.")
        if not reasons:
            reasons.append("Model predicts failure without a specific rule firing.")
            actions.append("Investigate conditions; run diagnostics and monitor closely.")
    else:
        reasons.append("Operating conditions appear within safe limits.")
        actions.append("Continue monitoring; follow standard inspection schedule.")
    return reasons, actions

def indicator_bar_chart(intensities: dict):
    labels = ["HDF", "PWF", "OSF", "TWF"]
    vals   = [intensities[k]*100 for k in labels]
    colors = []
    for v in vals:
        colors.append("#86efac" if v < 33 else ("#fde68a" if v < 66 else "#fecaca"))
    fig = go.Figure(go.Bar(
        x=vals, y=labels, orientation="h",
        marker_color=colors, text=[f"{v:.0f}%" for v in vals],
        textposition="outside"
    ))
    fig.update_layout(
        title="Indicator Intensities",
        xaxis=dict(range=[0,100], title="Intensity (0–100%)"),
        yaxis=dict(title=""),
        height=320, margin=dict(l=10, r=10, t=40, b=10)
    )
    return fig

# ── UI ───────────────────────────────────────────────────
st.set_page_config(page_title="Machine Failure Predictor", page_icon="🛠️", layout="wide")
st.title("Machine Failure Risk")

with st.expander("Enter operating parameters", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        t    = st.selectbox("Type", ["L", "M", "H"])
        air  = st.number_input("Air temperature [K]", 250.0, 400.0, 300.0)
        rpm  = st.number_input("Rotational speed [rpm]", 0, 3000, 1500)
    with col2:
        proc = st.number_input("Process temperature [K]", 250.0, 450.0, 310.0)
        tor  = st.number_input("Torque [Nm]", 0.0, 100.0, 40.0)
        wear = st.number_input("Tool wear [min]", 0, 300, 120)

row = pd.DataFrame([{
    "Type": t,
    "Air temperature [K]": air,
    "Process temperature [K]": proc,
    "Rotational speed [rpm]": rpm,
    "Torque [Nm]": tor,
    "Tool wear [min]": wear,
    "Temp diff [K]": proc - air,
}])

if st.button("Predict"):
    # Model probability & decision
    prob = float(pipe.predict_proba(row)[0, 1])
    fail = prob >= decision_threshold
    risk_pct = int(round(prob * 100))
    flags = rule_indicators(row)

    # Layout: gauge left, status/indicators right
    cg, cr = st.columns([0.6, 0.4])

    # Left: Gauge
    with cg:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_pct,
            number={'suffix': " %", 'font': {'size': 48}},
            title={'text': "⚠️ Risk" if fail else "🟢 Risk"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar':  {'color': "#0f172a"},
                'steps': [
                    {'range': [0, 50],  'color': "#DCFCE7"},
                    {'range': [50, 75], 'color': "#FEF9C3"},
                    {'range': [75, 100],'color': "#FEE2E2"},
                ],
                'threshold': {
                    'line': {'color': "#DC2626", 'width': 4},
                    'thickness': 0.75,
                    'value': risk_pct
                }
            }
        ))
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)

    # Right: chips + reason/solution
    with cr:
        st.markdown(
            f"""<div style="margin:6px 0 10px 0;">
                <span style="
                  display:inline-block;padding:8px 14px;border-radius:9999px;
                  background:{'#DCFCE7' if not fail else '#FEE2E2'};
                  color:{'#137333' if not fail else '#A50E0E'};
                  border:1px solid {'#BBF7D0' if not fail else '#FECACA'};
                  font-weight:800;">
                  {'✅ Safe — Risk '+str(risk_pct) if not fail else '⚠️ Fail — Risk '+str(risk_pct)}
                </span>
            </div>""",
            unsafe_allow_html=True
        )

        st.markdown("**Indicators:**")
        st.markdown(
            chip(f"HDF (ΔT={flags['dT']:.1f} K)", ok=not flags["flags"]["HDF"]) +
            chip(f"PWF (torque×rpm={int(flags['power_index'])})", ok=not flags["flags"]["PWF"]) +
            chip("OSF (wear×torque)", ok=not flags["flags"]["OSF"]) +
            chip(f"TWF (wear={int(flags['wear'])})", ok=not flags["flags"]["TWF"]),
            unsafe_allow_html=True
        )

        st.markdown("### Failure Reason & Immediate Solution")
        reasons, actions = reason_and_solution(flags, fail)
        st.markdown(f"**Reason:** {reasons[0]}")
        st.markdown(f"**Immediate Solution:** {actions[0]}")
        if len(reasons) > 1:
            st.info("Additional factors:\n" + "\n".join("• " + r for r in reasons[1:]))

    # Extra chart: indicator intensities
    st.plotly_chart(
        indicator_bar_chart(flags["intensities"]),
        use_container_width=True
    )

    st.caption(
        "Indicators use data-driven thresholds (if indicator_thresholds.json is present). "
        "Model decision uses your trained classifier; adjust cutoffs as needed."
    )
