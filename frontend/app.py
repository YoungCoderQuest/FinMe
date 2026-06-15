import streamlit as st
import requests

st.set_page_config(page_title="FinMe Demo", layout="wide")
st.title("FinMe Market Memory Demo")

st.write("This is a starter frontend for your market analysis chatbot.")

if st.button("Check backend health"):
    try:
        resp = requests.get("http://backend:8000/health", timeout=5)
        st.write("Backend response:", resp.json())
    except Exception as e:
        st.error(f"Backend check failed: {e}")

st.markdown("---")

st.subheader("Example prompt")
st.write(
    "Ask a question here after you wire the backend to Qdrant, Neo4j, and your market data sources."
)
