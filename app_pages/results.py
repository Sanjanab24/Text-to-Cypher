import json

import altair as alt
import pandas as pd
import streamlit as st

from app_utils import RESULTS


@st.cache_data
def numbers():
    unseen = {k: v["overall"]["exec_acc"]
              for k, v in json.load(open(RESULTS / "compare_test_unseen_1000_eg.json")).items()}

    def financial(name):
        rows = json.load(open(RESULTS / f"{name}.json", encoding="utf-8"))
        return round(100 * sum(r["gat"]["exec_guided"] for r in rows) / len(rows), 1)

    return (unseen["baseline_t5:test_unseen"], unseen["gat_t5:test_unseen.sample1000.eg"],
            financial("aml_check"), financial("aml_check_adapted"))


def bars(data, colors, alt_text):
    chart = (alt.Chart(pd.DataFrame(data, columns=["model", "accuracy"]))
             .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, size=70)
             .encode(x=alt.X("model:N", sort=None, title=None, axis=alt.Axis(labelAngle=0, labelFontSize=13)),
                     y=alt.Y("accuracy:Q", title="Correct answers (%)", scale=alt.Scale(domain=[0, 100])),
                     color=alt.Color("model:N", sort=None, scale=alt.Scale(range=colors), legend=None),
                     tooltip=["model", alt.Tooltip("accuracy:Q", format=".1f")]))
    labels = chart.mark_text(dy=-10, fontSize=16, fontWeight="bold").encode(text=alt.Text("accuracy:Q", format=".1f"),
                                                                             color=alt.value("#52514e"))
    st.altair_chart((chart + labels).properties(height=300), alt=alt_text)


base, ours, zero, adapted = numbers()
left, right = st.columns(2, gap="large")
with left.container(border=True, height="stretch"):
    st.markdown("**New databases** (never seen in training)")
    bars([("Baseline", base), ("Our model (GAT)", ours)], ["#9a9993", "#2a78d6"],
         f"Baseline {base:.1f}% vs our model {ours:.1f}% on new databases")
    st.caption(f"1,000 test questions. Graph attention (linking words to the schema) plus the execution check "
               f"adds +{ours - base:.1f} points.")
with right.container(border=True, height="stretch"):
    st.markdown("**Financial transactions** (our model, before and after financial training)")
    bars([("Before", zero), ("After", adapted)], ["#9a9993", "#1baf7a"],
         f"Our model on financial questions: {zero:.1f}% before, {adapted:.1f}% after financial training")
    st.caption(f"55 financial test questions. Training on ~1,000 synthetic financial questions adds "
               f"+{adapted - zero:.1f} points.")
