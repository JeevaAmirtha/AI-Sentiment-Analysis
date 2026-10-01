import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sentiment import analyze_sentiment


# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Sentiment Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

.hero {
    padding: 30px;
    border-radius: 18px;
    text-align: center;
    margin-bottom: 25px;
    border: 1px solid rgba(128,128,128,0.2);
}

.hero h1 {
    font-size: 42px;
    margin-bottom: 8px;
}

.hero p {
    font-size: 17px;
    opacity: 0.75;
}

.metric-card {
    padding: 20px;
    border-radius: 15px;
    border: 1px solid rgba(128,128,128,0.2);
    text-align: center;
}

.metric-title {
    font-size: 14px;
    opacity: 0.7;
}

.metric-value {
    font-size: 30px;
    font-weight: bold;
}

.section-title {
    font-size: 25px;
    font-weight: 600;
    margin-top: 15px;
}

.info-box {
    padding: 18px;
    border-radius: 12px;
    border: 1px solid rgba(128,128,128,0.2);
}

.footer {
    text-align: center;
    opacity: 0.6;
    margin-top: 40px;
    padding: 20px;
}

</style>
""", unsafe_allow_html=True)


# ---------------- SIDEBAR ----------------
with st.sidebar:

    st.title("📊 Sentiment AI")

    st.markdown("---")

    st.markdown("### About")

    st.write(
        "An AI-powered sentiment analysis application "
        "that analyzes customer reviews using NLP and "
        "a pretrained BERT-based Transformer model."
    )

    st.markdown("---")

    st.markdown("### Technologies")

    st.write("""
    • Python  
    • Streamlit  
    • spaCy  
    • BERT  
    • Transformers  
    • Pandas  
    • Matplotlib
    """)

    st.markdown("---")

    st.markdown("### Features")

    st.write("""
    ✓ Single text analysis  
    ✓ CSV batch analysis  
    ✓ Sentiment classification  
    ✓ Confidence score  
    ✓ Data visualization  
    ✓ Download results
    """)


# ---------------- HERO SECTION ----------------
st.markdown("""
<div class="hero">

<h1>🤖 AI Sentiment Analysis</h1>

<p>
Analyze customer reviews using Natural Language Processing
and a pretrained BERT-based Transformer model.
</p>

</div>
""", unsafe_allow_html=True)


# ---------------- TABS ----------------
tab1, tab2 = st.tabs([
    "💬 Single Text Analysis",
    "📁 CSV Batch Analysis"
])


# =========================================================
# SINGLE TEXT ANALYSIS
# =========================================================

with tab1:

    st.markdown(
        '<div class="section-title">Analyze a Review</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Enter a customer review below to identify its sentiment."
    )

    text = st.text_area(
        "Customer Review",
        placeholder="Example: I really love this product. The quality is amazing!",
        height=150
    )

    if st.button(
        "🔍 Analyze Sentiment",
        use_container_width=True
    ):

        if text.strip():

            with st.spinner("Analyzing sentiment..."):

                sentiment, confidence = analyze_sentiment(text)

            st.success("Analysis completed successfully!")

            col1, col2 = st.columns(2)

            with col1:

                st.markdown("""
                <div class="metric-card">
                <div class="metric-title">Detected Sentiment</div>
                <div class="metric-value">{}</div>
                </div>
                """.format(sentiment), unsafe_allow_html=True)

            with col2:

                st.markdown("""
                <div class="metric-card">
                <div class="metric-title">Confidence Score</div>
                <div class="metric-value">{:.2%}</div>
                </div>
                """.format(confidence), unsafe_allow_html=True)

        else:

            st.warning("Please enter a review first.")


# =========================================================
# CSV BATCH ANALYSIS
# =========================================================

with tab2:

    st.markdown(
        '<div class="section-title">Batch Sentiment Analysis</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Upload a CSV file containing a column named "
        "**review** to analyze multiple reviews."
    )

    uploaded_file = st.file_uploader(
        "Upload CSV File",
        type=["csv"]
    )

    if uploaded_file:

        df = pd.read_csv(uploaded_file)

        if "review" not in df.columns:

            st.error(
                "CSV must contain a column named 'review'."
            )

        else:

            st.success(
                f"File uploaded successfully — {len(df)} reviews found."
            )

            if st.button(
                "🚀 Analyze All Reviews",
                use_container_width=True
            ):

                sentiments = []
                confidences = []

                progress_bar = st.progress(0)

                for i, review in enumerate(df["review"]):

                    sentiment, confidence = analyze_sentiment(
                        str(review)
                    )

                    sentiments.append(sentiment)
                    confidences.append(confidence)

                    progress_bar.progress(
                        (i + 1) / len(df)
                    )

                df["Sentiment"] = sentiments
                df["Confidence"] = confidences

                st.session_state["results"] = df


# =========================================================
# RESULTS DASHBOARD
# =========================================================

if "results" in st.session_state:

    results = st.session_state["results"]

    st.markdown("---")

    st.markdown(
        '<div class="section-title">📈 Analysis Dashboard</div>',
        unsafe_allow_html=True
    )

    total = len(results)

    positive = (results["Sentiment"] == "Positive").sum()
    negative = (results["Sentiment"] == "Negative").sum()
    neutral = (results["Sentiment"] == "Neutral").sum()

    average_confidence = results["Confidence"].mean()


    # ---------------- METRICS ----------------

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Total Reviews", total)

    with col2:
        st.metric("Positive", positive)

    with col3:
        st.metric("Negative", negative)

    with col4:
        st.metric("Neutral", neutral)

    with col5:
        st.metric(
            "Avg. Confidence",
            f"{average_confidence:.1%}"
        )


    st.markdown("### 📊 Sentiment Distribution")

    col1, col2 = st.columns(2)


    # ---------------- BAR CHART ----------------

    with col1:

        sentiment_counts = results["Sentiment"].value_counts()

        st.bar_chart(sentiment_counts)


    # ---------------- PIE CHART ----------------

    with col2:

        fig, ax = plt.subplots()

        ax.pie(
            sentiment_counts.values,
            labels=sentiment_counts.index,
            autopct="%1.1f%%",
            startangle=90
        )

        ax.set_title("Sentiment Distribution")

        st.pyplot(fig)


    # ---------------- RESULTS TABLE ----------------

    st.markdown("### 📋 Detailed Results")

    st.dataframe(
        results,
        use_container_width=True,
        hide_index=True
    )


    # ---------------- DOWNLOAD ----------------

    csv = results.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇️ Download Analysis Results",
        data=csv,
        file_name="sentiment_analysis_results.csv",
        mime="text/csv",
        use_container_width=True
    )


# ---------------- FOOTER ----------------

st.markdown("""
<div class="footer">

AI Sentiment Analysis • Built with Python, Streamlit & BERT

</div>
""", unsafe_allow_html=True)