import streamlit as st

from news_rag.factory import build_pipeline

st.set_page_config(page_title="AI News Research Assistant", layout="wide")
st.title("AI News Research Assistant")
st.caption("Two-stage RAG pipeline using OpenAI embeddings and Pinecone")

question = st.text_input("Ask a question about the indexed news dataset")
top_k = st.slider("Number of retrieved articles", min_value=1, max_value=10, value=3)

if st.button("Ask") and question:
    with st.spinner("Retrieving articles and generating grounded answer..."):
        pipeline = build_pipeline()
        answer = pipeline.answer(question, top_k=top_k)
        st.subheader("Answer")
        st.write(answer)
