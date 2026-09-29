import streamlit as st
import pandas as pd
from sentiment import analyze_sentiment
import matplotlib.pyplot as plt
st.set_page_config(
    page_title="AI Sentiment Analyzer",
    page_icon="💬",
    layout="wide"
)
# Custom CSS
st.markdown("""
<style>

.main-title {
    font-size: 42px;
    font-weight: 700;
    text-align: center;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    margin-bottom: 30px;
}

.section-title {
    font-size: 25px;
    font-weight: 600;
    margin-top: 20px;
}

</style>
""", unsafe_allow_html=True)
# Sidebar
with st.sidebar:
    st.title("💬 Sentiment AI")
    st.markdown("---")

    st.write("### About")
    st.write(
        "AI-powered sentiment analysis using "
        "spaCy and a pretrained BERT-based model."
    )

    st.markdown("---")

    st.write("### Technologies")
    st.write("🐍 Python")
    st.write("🧠 BERT")
    st.write("🔤 spaCy")
    st.write("📊 Pandas")
    st.write("🎨 Streamlit")


# Main title
st.markdown(
    '<div class="main-title">💬 AI Sentiment Analysis</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Analyze reviews using NLP and a pretrained BERT-based model'
    '</div>',
    unsafe_allow_html=True
)

st.markdown("---")


# Single Text Analysis
st.header("📝 Analyze Individual Text")

text = st.text_area(
    "Enter a review, comment, or feedback:",
    height=150,
    placeholder="Example: The product quality is amazing!"
)

if st.button("🔍 Analyze Sentiment", use_container_width=True):

    if not text.strip():
        st.warning("Please enter some text.")

    else:
        with st.spinner("Analyzing sentiment..."):
            sentiment, confidence = analyze_sentiment(text)

        st.markdown("---")
        st.subheader("📊 Analysis Result")

        col1, col2 = st.columns(2)

        with col1:
            if sentiment == "Positive":
                st.success(f"😊 {sentiment}")
            elif sentiment == "Negative":
                st.error(f"😞 {sentiment}")
            else:
                st.info(f"😐 {sentiment}")

        with col2:
            st.metric(
                "Confidence",
                f"{confidence:.2%}"
            )


# CSV Analysis
st.markdown("---")
st.header("📂 Analyze Multiple Reviews")

uploaded_file = st.file_uploader(
    "Upload a CSV file containing reviews",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Uploaded Data")
    st.dataframe(df, use_container_width=True)

    if "review" not in df.columns:

        st.error(
            "CSV must contain a column named 'review'."
        )

    else:

        if st.button(
            "🤖 Analyze All Reviews",
            use_container_width=True
        ):

            sentiments = []
            confidences = []

            progress = st.progress(0)

            total = len(df)

            for i, review in enumerate(df["review"]):

                sentiment, confidence = analyze_sentiment(
                    str(review)
                )

                sentiments.append(sentiment)
                confidences.append(confidence)

                progress.progress(
                    (i + 1) / total
                )

            df["Sentiment"] = sentiments
            df["Confidence"] = confidences

            st.success(
                "✅ All reviews analyzed successfully!"
            )

            st.subheader("📊 Analysis Results")

            st.dataframe(
                df,
                use_container_width=True
            )
                        # Download Results
            csv = df.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name="sentiment_analysis_results.csv",
                mime="text/csv",
                use_container_width=True
            )
            # Summary
            st.subheader("📈 Sentiment Summary")

            positive = (df["Sentiment"] == "Positive").sum()
            negative = (df["Sentiment"] == "Negative").sum()
            neutral = (df["Sentiment"] == "Neutral").sum()
                        # Sentiment Chart Data
            chart_data = pd.DataFrame({
                "Sentiment": ["Positive", "Negative", "Neutral"],
                "Count": [positive, negative, neutral]
            })

            st.markdown("---")
            st.subheader("📊 Sentiment Distribution")

            st.bar_chart(
                chart_data.set_index("Sentiment")
            )

            st.subheader("🥧 Sentiment Overview")

            st.dataframe(
                chart_data,
                use_container_width=True
            )
                        # Pie Chart
            st.subheader("🥧 Sentiment Distribution")

            fig, ax = plt.subplots()

            ax.pie(
                chart_data["Count"],
                labels=chart_data["Sentiment"],
                autopct="%1.1f%%",
                startangle=90
            )

            ax.set_title("Sentiment Distribution")

            st.pyplot(fig)

            # Average Confidence
            st.subheader("🎯 Model Confidence")

            average_confidence = df["Confidence"].mean()

            st.metric(
                "Average Confidence",
                f"{average_confidence:.2%}"
            )
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("😊 Positive", positive)

            with col2:
                st.metric("😞 Negative", negative)

            with col3:
                st.metric("😐 Neutral", neutral)
                            # Dashboard Metrics
            total_reviews = len(df)

            st.markdown("---")
            st.subheader("📊 Dashboard Overview")

            metric1, metric2, metric3, metric4 = st.columns(4)

            with metric1:
                st.metric(
                    "📋 Total Reviews",
                    total_reviews
                )

            with metric2:
                st.metric(
                    "😊 Positive",
                    positive
                )

            with metric3:
                st.metric(
                    "😞 Negative",
                    negative
                )

            with metric4:
                st.metric(
                    "😐 Neutral",
                    neutral
                )