import streamlit as st
import tempfile

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

# Transformers (local model)
from transformers import pipeline

# Streamlit page config
st.set_page_config(page_title="Chat with PDF")
st.title(" Chat with your PDF")

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")


#  Cache embedding + vector store (important for speed)
@st.cache_resource
def create_vectorstore(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.from_documents(chunks, embeddings)
    return db


#  Load local LLM (cached)
@st.cache_resource
def load_llm():
    pipe = pipeline(
    "text-generation",
    model="google/flan-t5-base",
    max_length=512
)
    return HuggingFacePipeline(pipeline=pipe)


if uploaded_file is not None:
    st.info("Processing PDF...")

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        file_path = tmp_file.name

    try:
        db = create_vectorstore(file_path)
    except Exception as e:
        st.error(f"Failed to process PDF: {e}")
        st.stop()

    retriever = db.as_retriever(search_kwargs={"k": 3})

    llm = load_llm()

    # Prompt Template
    prompt = PromptTemplate.from_template(
        "Answer the question based only on the context:\n{context}\n\nQuestion: {question}"
    )

    # Helper function
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # RAG chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    st.success("PDF processed! Ask your question 👇")

    query = st.text_input("Ask a question from the PDF")

    if query:
        with st.spinner("Generating answer..."):
            response = rag_chain.invoke(query)

        st.subheader("Answer:")
        st.write(response)

        # Show sources
        st.subheader("Source Chunks:")
        results = db.similarity_search(query, k=2)
        for i, r in enumerate(results, 1):
            st.markdown(f"**Chunk {i}:**")
            st.write(r.page_content)
            st.write("---")
