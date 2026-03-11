import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import requests

# 🔑 HuggingFace API Token (Optional – works even if left empty)
HF_API_TOKEN = ""

API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-large"
headers = {"Authorization": f"Bearer {HF_API_TOKEN}"} if HF_API_TOKEN else {}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

# ---------------- UI ----------------

st.set_page_config(page_title="Talking Rabbitt MCP", layout="wide")

st.title("🐰 Talking Rabbitt – Conversational Business Intelligence")
st.markdown("Upload your business dataset and ask executive-level questions.")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

# ---------------- DATA LOAD ----------------

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Data Preview")
    st.dataframe(df.head())

    question = st.text_input("💬 Ask a business question:")

    if question:

        with st.spinner("Analyzing business performance..."):

            answer = ""
            q = question.lower()
            
            # ---------------- TRY LLM (Optional) ----------------
            if HF_API_TOKEN:
                try:
                    data_sample = df.head(30).to_string()

                    prompt = f"""
                    You are a senior business strategy consultant.
                    Provide a concise executive insight (2-3 sentences).

                    Dataset:
                    {data_sample}

                    Question: {question}
                    """

                    output = query({"inputs": prompt})

                    if isinstance(output, list):
                        answer = output[0]["generated_text"]
                    else:
                        raise Exception("Model not ready")

                except:
                    answer = ""

            # ---------------- LOCAL INTELLIGENCE FALLBACK ----------------
            if not answer:

                if "highest" in q and "revenue" in q and "Region" in df.columns:
                    result = df.groupby("Region")["Revenue"].sum()
                    top_region = result.idxmax()
                    top_value = result.max()

                    answer = (
                        f"{top_region} generated the highest revenue at ${top_value:,.0f}. "
                        "This region is currently leading overall performance and represents a strong growth driver."
                    )

                elif "total" in q and "revenue" in q:
                    total = df["Revenue"].sum()
                    answer = (
                        f"Total revenue across all regions is ${total:,.0f}. "
                        "This reflects cumulative business performance within the dataset."
                    )

                elif "units" in q and "Units_Sold" in df.columns:
                    total_units = df["Units_Sold"].sum()
                    answer = (
                        f"A total of {total_units:,.0f} units were sold. "
                        "This indicates overall market demand across regions."
                    )

                elif "quarter" in q and "Quarter" in df.columns:
                    summary = df.groupby("Quarter")["Revenue"].sum()
                    answer = "Revenue by quarter:\n\n"
                    for quarter, value in summary.items():
                        answer += f"{quarter}: ${value:,.0f}\n"

                else:
                    answer = "Insight generated based on available dataset metrics."

        # ---------------- DISPLAY INSIGHT ----------------

        st.subheader("📊 Executive Insight")
        st.success(answer)

        # ---------------- SMART VISUALIZATION ----------------

        st.subheader("📈 Smart Visualization")

        fig, ax = plt.subplots(figsize=(8, 4))
        plt.tight_layout()

        if "revenue" in q and "Region" in df.columns:
            df.groupby("Region")["Revenue"].sum().plot(kind="bar", ax=ax)
            ax.set_title("Revenue by Region")

        elif "units" in q and "Region" in df.columns:
            df.groupby("Region")["Units_Sold"].sum().plot(kind="bar", ax=ax)
            ax.set_title("Units Sold by Region")

        elif "quarter" in q and "Quarter" in df.columns:
            df.groupby("Quarter")["Revenue"].sum().plot(kind="bar", ax=ax)
            ax.set_title("Revenue by Quarter")

        else:
            numeric_cols = df.select_dtypes(include=["number"]).columns
            if len(numeric_cols) > 0:
                df[numeric_cols[0]].plot(kind="bar", ax=ax)
                ax.set_title(f"{numeric_cols[0]} Distribution")

        st.pyplot(fig)
