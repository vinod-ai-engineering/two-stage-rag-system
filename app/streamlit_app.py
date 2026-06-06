import streamlit as st

from news_rag.factory import build_pipeline

st.set_page_config(page_title="AI News Research Assistant", layout="wide")
st.title("AI News Research Assistant")
st.caption("Two-stage RAG demo using OpenAI embeddings and Pinecone vector search")

question = st.text_input("Ask a news research question", value="What happened with Obama?")
top_k = st.slider("Top K articles", min_value=1, max_value=10, value=3)

if st.button("Ask"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Retrieving articles and generating answer..."):
            pipeline = build_pipeline()
            answer = pipeline.answer(question, top_k=top_k)
        st.subheader("Answer")
        st.write(answer)
