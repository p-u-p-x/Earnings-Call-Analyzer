import streamlit as st
import requests
import plotly.graph_objects as go

BACKEND = "http://backend:8000"

st.set_page_config(page_title="Earnings Call Analyzer", layout="wide")
st.title("Earnings Call Analyzer")
st.caption("Upload an earnings call transcript PDF to analyze")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file and st.button("Analyze Transcript"):
    with st.spinner("Analyzing... this takes 1 to 2 minutes"):
        response = requests.post(
            f"{BACKEND}/analyze",
            files={"file": (uploaded_file.name, uploaded_file, "application/pdf")}
        )
        if response.status_code == 200:
            st.session_state["results"] = response.json()
        else:
            st.error(f"Backend error: {response.text}")

if "results" in st.session_state:
    data = st.session_state["results"]
    tab1, tab2, tab3, tab4 = st.tabs([
        "Companies and People", "Sentiment", "Summary", "Ask a Question"
    ])

    with tab1:
        st.subheader("Named Entities")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Companies**")
            for c in data["ner"]["companies"]:
                st.write(f"- {c}")
        with col2:
            st.write("**People**")
            for p in data["ner"]["people"]:
                st.write(f"- {p}")

    with tab2:
        st.subheader("Sentiment Analysis")
        avg = data["sentiment"]["average"]
        scores = data["sentiment"]["scores"]
        label = "Positive" if avg > 0.2 else ("Negative" if avg < -0.2 else "Neutral")
        st.metric("Overall Sentiment", label, f"Score: {avg:.3f}")
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=scores, mode="lines+markers", name="Sentiment"))
        fig.add_hline(y=0, line_dash="dash", line_color="black")
        fig.update_layout(title="Sentiment Across Sentences",
                         xaxis_title="Sentence", yaxis_title="Score")
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Executive Summary")
        st.write(data["summary"])
        st.caption(f"Analyzed {data['characters']:,} characters")

    with tab4:
        st.subheader("Ask a Question")
        question = st.text_input("Ask anything about this transcript")
        if st.button("Ask") and question:
            with st.spinner("Thinking..."):
                res = requests.post(
                    f"{BACKEND}/ask",
                    json={"question": question, "context": data["summary"]}
                )
                if res.status_code == 200:
                    st.write(res.json()["answer"])
