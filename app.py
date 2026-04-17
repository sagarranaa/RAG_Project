from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.schema import Document

# ========== STEP 1: LOAD PDF ==========
try:
    loader = PyPDFLoader("sample.pdf")
    documents = loader.load()
    print("Documents loaded:", len(documents))
except Exception as e:
    print("Error loading PDF:", e)
    documents = []

# Debug: show content
for i, doc in enumerate(documents):
    print(f"\nDoc {i} preview:\n", repr(doc.page_content[:200]))

# ========== STEP 2: CLEAN EMPTY TEXT ==========
documents = [doc for doc in documents if doc.page_content.strip()]
print("\nAfter cleaning empty docs:", len(documents))

# 🚨 Fallback if PDF is empty
if len(documents) == 0:
    print("\n⚠️ PDF has no readable text. Using fallback sample data...\n")
    documents = [
        Document(
            page_content="RAG stands for Retrieval Augmented Generation. It helps large language models answer questions using external data."
        )
    ]

# ========== STEP 3: SPLIT INTO CHUNKS ==========
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

docs = text_splitter.split_documents(documents)
print("Chunks created:", len(docs))

# 🚨 Safety check
if len(docs) == 0:
    raise ValueError("No text chunks created. Check your PDF content.")

# ========== STEP 4: EMBEDDINGS ==========
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ========== STEP 5: VECTOR DB ==========
db = Chroma.from_documents(docs, embedding)

# ========== STEP 6: LOAD LLM ==========
llm = OllamaLLM(model="llama3")

# ========== STEP 7: QUERY LOOP ==========
retriever = db.as_retriever()

while True:
    query = input("\nAsk something (or type 'exit'): ")

    if query.lower() == "exit":
        break

    # Retrieve docs
    relevant_docs = retriever.invoke(query)
    print("Retrieved docs:", len(relevant_docs))

    # Combine context
    context = " ".join([doc.page_content for doc in relevant_docs])

    # Generate answer
    response = llm.invoke(
        f"Answer ONLY from this context:\n{context}\n\nQuestion: {query}"
    )

    print("\nAnswer:\n", response)