import pandas as pd
import streamlit as st
from sklearn.metrics import precision_recall_curve

st.set_page_config(page_title="Incident Risk Dashboard", layout="wide")
st.title("Nexlyra Incident Risk (synthetic data)")
st.warning("Synthetic dataset. Do not use for real alerting.")

df = pd.read_csv("reports/test_predictions.csv")
y, p = df["incident_next_60m"], df["risk"]

budget = st.slider("Alert budget (per 1,000 events)", 5, 200, 50)
k = int(len(df) * budget / 1000)
top = df.nlargest(k, "risk")
c1, c2, c3 = st.columns(3)
c1.metric("Recall at budget", f"{top['incident_next_60m'].sum() / max(y.sum(), 1):.1%}")
c2.metric("Precision at budget", f"{top['incident_next_60m'].mean():.1%}")
c3.metric("False alerts / 1,000", f"{(k - top['incident_next_60m'].sum()) / len(df) * 1000:.1f}")

prec, rec, _ = precision_recall_curve(y, p)
st.line_chart(pd.DataFrame({"recall": rec, "precision": prec}).set_index("recall"))
st.subheader("Top risky events")
st.dataframe(top[["event_id", "event_timestamp", "risk"]].head(50))
