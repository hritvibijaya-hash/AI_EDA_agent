import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import streamlit as st

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain.agents import create_agent

st.set_page_config(page_title="AI Powered Data Analyst Agent", layout="wide")
st.title("🤖 AI-Powered Data Analyst Agent")
st.write("Upload your dataset or use the default one to perform automated EDA, generate charts (univariate, bivariate, multivariate), and chat with your data!")

GOOGLE_API_KEY = st.sidebar.text_input("Enter Google API Key", type="password")
GROQ_API_KEY = st.sidebar.text_input("Enter Groq API Key", type="password")

if not GOOGLE_API_KEY or not GROQ_API_KEY:
    st.warning("Please enter both Google and Groq API keys in the sidebar to proceed.")
    st.stop()

gemini_llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    google_api_key=GOOGLE_API_KEY
)

groq_llm = ChatGroq(
    model="qwen/qwen3.6-27b",
    api_key=GROQ_API_KEY
)

def temp_tool():
    """This is just a dummy tool"""
    return "Hello world"

agent = create_agent(
    model=gemini_llm,
    tools=[temp_tool]
)

uploaded_file = st.file_uploader("Upload CSV, XLS, or XLSX file", type=["csv", "xls", "xlsx"])

@st.cache_data
def load_data(file_path_or_buffer):
    if isinstance(file_path_or_buffer, str):
        if file_path_or_buffer.endswith('.csv'):
            return pd.read_csv(file_path_or_buffer)
        else:
            return pd.read_excel(file_path_or_buffer)
    else:
        try:
            return pd.read_csv(file_path_or_buffer)
        except Exception:
            file_path_or_buffer.seek(0)
            return pd.read_excel(file_path_or_buffer)

if uploaded_file is not None:
    df = load_data(uploaded_file)
    st.success("File uploaded successfully!")
else:
    default_url = 'https://raw.githubusercontent.com/axisgras-hash/DATASETS/refs/heads/main/Superstore.csv'
    st.info("No file uploaded. Using default Superstore dataset.")
    df = load_data(default_url)

st.subheader("Dataset Preview")
st.dataframe(df.head())

if st.button("Run Automated EDA & Generate Charts"):
    with st.spinner("Agent is analyzing dataset and writing python functions for EDA..."):
        try:
            sample_df = df.sample(min(5, len(df)))
            prompt = f"""You are a data analyst. Perform basic eda python single function perform_eda
            code and give all required analysis like missing values and columns.
            Data frame sample : {sample_df}
            data stats: {sample_df.describe()}"""

            response = agent.invoke({'messages': [{'role': 'user', 'content': prompt}]})
            ans = response["messages"][-1].content[-1]['text']
            code = ans.split("```")[1]
            if code.startswith("python"):
                code = code[6:]

            with open('basic_eda.py', 'w') as f:
                f.write(code)

            advance_prompt = """give detailed prompt for advance data analysis, which must include
            describe, corr, univariate numerical and object column analysis, bivariate analysis,
            time series if any date column given, multivariate analysis to perform different col like
            example sales, region, segment using bar plot with hue, give code with strict python
            and module code with pip install for any unknown new module if required"""

            response_adv = agent.invoke({'messages': [{'role': 'user', 'content': advance_prompt}]})
            system_prompt_model = response_adv["messages"][-1].content[-1]['text']

            new_prompt = """Give Python advance_eda.py file with every code inside a single function eda_by_ai
            and no need to load file, df is already loaded, starts with using df
            and any notes with comment""" + system_prompt_model

            response_final = agent.invoke({'messages': [{'role': 'user', 'content': new_prompt}]})
            ans_final = response_final["messages"][-1].content[-1]['text']
            code_final = ans_final.split("```")[1]
            if code_final.startswith("python"):
                code_final = code_final[6:]

            with open('advance_eda.py', 'w') as f:
                f.write(code_final)

            st.success("EDA code generated and saved successfully!")
            
        except Exception as e:
            st.error(f"Error during code generation: {e}")

    st.subheader("Basic & Advanced EDA Results")
    try:
        from basic_eda import perform_eda
        st.write("### Basic EDA Summary")
        st.write(perform_eda(df) if callable(perform_eda) else "Basic EDA module loaded.")
    except Exception as e:
        st.warning(f"Could not automatically execute basic_eda: {e}")

    try:
        from advance_eda import eda_by_ai
        st.write("### Advanced EDA & Charts")
        fig, axes = plt.subplots(figsize=(10, 5))
        eda_by_ai(df)
        st.pyplot(plt)
        plt.clf()
    except Exception as e:
        st.warning(f"Could not automatically execute advance_eda function, showing standard plots: {e}")
        st.write("#### Univariate & Multivariate Quick Charts")
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if numeric_cols:
            fig, ax = plt.subplots()
            sns.histplot(df[numeric_cols[0]], kde=True, ax=ax)
            st.pyplot(fig)
            plt.clf()

st.subheader("💬 Chat with Your Data")
user_query = st.text_input("Ask anything about your dataset (e.g., 'What is the total sales by region?'):")

if user_query:
    with st.spinner("Agent is generating code/answer for your query..."):
        chat_prompt = f"""Given the dataframe df with columns {list(df.columns)}, 
        answer the following user query by writing python code or giving a direct analytical response: {user_query}"""
        
        chat_response = agent.invoke({'messages': [{'role': 'user', 'content': chat_prompt}]})
        chat_ans = chat_response["messages"][-1].content[-1]['text']
        st.markdown(chat_ans)
